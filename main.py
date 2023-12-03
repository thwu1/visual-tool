import math
import time

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from utils import color_map, entropy_from_logits, logprobs_from_logits, openchat_template


class VisualizeLLM:
    def __init__(self, model_name, extra_generate_kwargs, chat_template, use_vllm=False, use_flash_attention_2=False) -> None:
        self.use_vllm = use_vllm
        if use_vllm:
            self.llm = LLM(model_name)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch.bfloat16).to(
                "cuda"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.extra_generate_kwargs = extra_generate_kwargs  # deprecated
        if self.extra_generate_kwargs.get("pad_token_id") is None:
            self.extra_generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat_template = chat_template

        max_new_token_slider = gr.Slider(minimum=1, maximum=8192, step=1, label="Max New Tokens", value=256)
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.01, label="Temperature", value=0.5)
        use_beam_search_box = gr.Checkbox(label="Use beam search", value=False)
        top_k_slider = gr.Slider(minimum=0, maximum=1000, step=1, label="Top K", value=0)
        top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, label="Top P", value=1)
        best_of_slider = gr.Slider(minimum=1, maximum=1000, step=1, label="Best Of, for beam search", value=1)
        return_prompt_box = gr.Checkbox(label="Return Prompt", value=False)

        self.iface = gr.Interface(
            fn=lambda *args: self.to_html_str(*self.gen_and_cal_prob(*args)),
            inputs=[
                "text",
                max_new_token_slider,
                temperature_slider,
                use_beam_search_box,
                top_k_slider,
                top_p_slider,
                best_of_slider,
                return_prompt_box,
            ],
            outputs=[gr.HTML("Probability", show_label=True), gr.HTML("Entropy", show_label=True), gr.HTML("Stats")],
            theme="default",
            title="Visualize LLMs",
            description=f"""    
            <p><b>Model name: {model_name}</b>. The color visualizes the log-probabilities of the generated tokens, from <b><span style='color:red;'>Red</span></b> (low probability) to <b><span style='color:green;'>Green</span></b> (high probability).</p>
            <ul>
                <li>Greedy decoding by calling <code>greedy_search()</code> if <code>best_of=1</code> and <code>use_beam_search=True</code></li>
                <li>Contrastive search by calling <code>contrastive_search()</code> if <code>penalty_alpha>0</code> and <code>top_k>1</code></li>
                <li>Multinomial sampling by calling <code>sample()</code> if <code>best_of=1</code> and <code>use_beam_search=False</code></li>
                <li>Beam-search decoding by calling <code>beam_search()</code> if <code>best_of>1</code> and <code>use_beam_search=True</code></li>
                <li>Beam-search multinomial sampling by calling <code>beam_sample()</code> if <code>best_of>1</code> and <code>use_beam_search=False</code></li>
                <li>Diverse beam-search decoding by calling <code>group_beam_search()</code>, if <code>best_of>1</code> and <code>num_beam_groups>1</code></li>
                <li>Constrained beam-search decoding by calling <code>constrained_beam_search()</code>, if <code>constraints!=None</code> or <code>force_words_ids!=None</code></li>
            </ul>
            """,
        )

    def get_logprob_and_entropy(self, input_ids, output_ids):
        """
        input_ids: [batch_size, seq_len]
        output_ids: [batch_size, seq_len]
        return: [{token_id: logprob}, ...]"""
        num_gen_tokens = output_ids.shape[-1] - input_ids.shape[-1]
        # print(num_gen_tokens)
        assert torch.eq(output_ids[0][: input_ids.shape[-1]], input_ids[0]).all()
        input_kwargs = {
            "input_ids": output_ids,
            "attention_mask": torch.ones_like(output_ids),
            "return_dict": True,
        }
        logits = self.llm(**input_kwargs).logits
        assert logits.shape[1] == input_ids.shape[-1] + num_gen_tokens
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_kwargs["input_ids"][:, 1:])
        entropys = entropy_from_logits(logits[:, :-1, :])
        logprob_ls = []
        entropy_ls = []
        for token, logprob, entropy in zip(output_ids[0][-num_gen_tokens:], logprobs[0][-num_gen_tokens:], entropys[0][-num_gen_tokens:]):
            logprob_ls.append({token.item(): logprob.item()})
            entropy_ls.append({token.item(): entropy.item()})
        return logprob_ls, entropy_ls

    def cal_stats(self, logprobs, entropys=None):
        """logprobs = [{token_id: logprob}, ...}]"""
        ls = []
        for logprob in logprobs:
            for token_id, logprob in logprob.items():
                ls.append(
                    {
                        "text": self.tokenizer.convert_ids_to_tokens(token_id).replace("▁", " ").replace("<0x0A>", "<|new_line|>"),
                        "prob": math.exp(logprob),
                    }
                )
        if entropys is not None:
            for pb, entropy in zip(ls, entropys):
                for token_id, ep in entropy.items():
                    pb["entropy"] = ep
        return ls

    def gen_and_cal_prob(self, text, max_new_tokens, temperature, use_beam_search, top_k, top_p, best_of, return_prompt):
        logs = "<b>Stats:</b><br>"
        input_text = [self.chat_template(text)]
        input_ids = self.tokenizer(input_text)["input_ids"]
        if self.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else -1,
                max_tokens=max_new_tokens,
                use_beam_search=use_beam_search,
                logprobs=1,
                stop_token_ids=[self.tokenizer.eos_token_id],
                best_of=best_of,
                skip_special_tokens=False,
            )

            start = time.time()
            outputs = self.llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)[0].outputs[0]
            generate_time = time.time() - start
            logs += f"generate time: {generate_time:.2f} s<br>"
            values = self.cal_stats(outputs.logprobs)
        else:
            input_ids = torch.tensor(input_ids).to("cuda")
            input_kwargs = {
                "input_ids": input_ids.to("cuda"),
                "attention_mask": torch.ones_like(input_ids).to("cuda"),
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "do_sample": not use_beam_search,
                "num_beams": best_of if use_beam_search else 1,
                "return_dict": True,
            }
            start = time.time()
            output_ids = self.llm.generate(**input_kwargs)
            generate_time = time.time() - start
            logs += f"generate time: {generate_time:.2f} s<br>"

            start = time.time()
            logprobs, entropys = self.get_logprob_and_entropy(input_ids, output_ids)
            logs += f"cal logs time: {time.time() - start:.2f} s<br>"
            values = self.cal_stats(logprobs=logprobs, entropys=entropys)

        logs += f"total logprob: {sum([math.log(v['prob']) for v in values]):.2f}, logprob/token: {sum([math.log(v['prob']) for v in values]) / len(values):.2f}<br>"
        if values[0].get("entropy") is not None:
            logs += f"total entropy: {sum([v['entropy'] for v in values]):.2f}, entropy/token: {sum([v['entropy'] for v in values]) / len(values):.2f}<br>"
        logs += f"num tokens: {len(values)}<br>"
        logs += f"num tokens per second: {len(values) / generate_time:.2f}<br>"

        return logs, values

    def to_html_str(self, logs: str, values) -> [str, str]:
        """values is a list of dictionaries like [{'text': 'Hello', 'prob': 0.5, 'entropy'}, ...]"""
        keys = [key for key in values[0].keys() if key != "text"]

        colored_text = [""] * len(keys)
        # add legends to colored_text
        for i, key in enumerate(keys):
            colored_text[i] += f"<strong>{key}:</strong><br>"
        log_info = logs + "<br><b>Legend:</b><br>"
        for item in values:
            text = item["text"] if item["text"] != "<|new_line|>" else " <br>"
            log_info += f"{text}: "
            for i, key in enumerate(keys):
                background_color = color_map(item[key], key)
                colored_text[i] += f"<span style='background-color: {background_color}; padding: 0px;'>{text}</span>"
                log_info += f"{key}: <span style='background-color: {background_color}; padding: 0px;'>{item[key]:.2f}</span>, "
            log_info += "<br>"
        return colored_text[0], colored_text[1], log_info

    def run(self):
        self.iface.launch(share=True)


if __name__ == "__main__":
    model_name = "openchat/openchat_3.5"
    # model_name = "lvwerra/gpt2-imdb"
    extra_generate_kwargs = {}
    server = VisualizeLLM(model_name, extra_generate_kwargs, openchat_template)
    server.run()
