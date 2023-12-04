import math
import time

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gc
from utils import color_map, entropy_from_logits, logprobs_from_logits, openchat_template
from dataclasses import dataclass
import tyro
from typing import Optional


class VisualizeLLM:
    def __init__(self, model_name, ref_model_name, extra_generate_kwargs, chat_template, use_flash_attention_2=False, low_cpu_mem_usage=True) -> None:
        self.model_name = model_name
        self.ref_model_name = ref_model_name
        self.use_flash_attention_2 = use_flash_attention_2
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self._restart()  # initialize self.llm and self.ref_llm

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.extra_generate_kwargs = extra_generate_kwargs  # deprecated
        if self.extra_generate_kwargs.get("pad_token_id") is None:
            self.extra_generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat_template = chat_template

        max_new_token_slider = gr.Slider(minimum=1, maximum=3072, step=1, label="Max New Tokens", value=256)
        temperature_slider = gr.Slider(minimum=0.01, maximum=1.5, step=0.01, label="Temperature", value=0.5)
        use_beam_search_box = gr.Checkbox(label="Use beam search", value=False)
        top_k_slider = gr.Slider(minimum=0, maximum=1000, step=1, label="Top K", value=0)
        top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, label="Top P", value=1)
        best_of_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Best Of, for beam search", value=1)
        return_prompt_box = gr.Checkbox(label="Return Prompt", value=False)

        self.iface = gr.Interface(
            fn=self.fn,
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
            outputs=[gr.HTML("Output", show_label=True), gr.HTML("Stats")],
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

    def fn(self, *args):
        return self.to_html_str(*self.gen_and_cal_prob(*args))

    def _clean_up(self):
        self.llm.to("cpu")
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        if self.ref_model_name is not None:
            self.ref_llm.to("cpu")
            del self.ref_llm
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

    def _restart(self):
        self.llm = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_flash_attention_2=self.use_flash_attention_2,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                device_map="cuda",
            )
            .eval()
            .requires_grad_(False)
        )
        if self.ref_model_name is not None:
            self.ref_llm = (
                AutoModelForCausalLM.from_pretrained(
                    self.ref_model_name,
                    use_flash_attention_2=self.use_flash_attention_2,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=self.low_cpu_mem_usage,
                    device_map="cuda",
                )
                .eval()
                .requires_grad_(False)
            )

    def get_logprob_and_entropy(self, llm, input_ids, output_ids):
        """
        Input:
        input_ids: [batch_size, seq_len]
        output_ids: [batch_size, seq_len]

        Return:
        values_ls: [{"token_id": token_id, "logprob": logprob}, ...]"""

        num_gen_tokens = output_ids.shape[-1] - input_ids.shape[-1]
        assert torch.eq(output_ids[0][: input_ids.shape[-1]], input_ids[0]).all()
        input_kwargs = {
            "input_ids": output_ids,
            "attention_mask": torch.ones_like(output_ids),
            "return_dict": True,
        }
        logits = llm(**input_kwargs).logits
        assert logits.shape[1] == input_ids.shape[-1] + num_gen_tokens
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_kwargs["input_ids"][:, 1:])
        entropys = entropy_from_logits(logits[:, :-1, :])
        values_ls = []
        for token_id, logprob, entropy in zip(output_ids[0][-num_gen_tokens:], logprobs[0][-num_gen_tokens:], entropys[0][-num_gen_tokens:]):
            values_ls.append({"token_id": token_id.item(), "logprob": logprob.item(), "entropy": entropy.item()})
        return values_ls

    def cal_stats(self, values_ls, ref_values_ls=None):
        """values_ls = [{"token_id": token_id, "logprob": logprob, "entropy": entropy}, ...]"""
        ls = []
        for values in values_ls:
            new_dict = {}
            for key, value in values.items():
                if key == "token_id":
                    new_dict["text"] = self.tokenizer.convert_ids_to_tokens(value).replace("▁", " ").replace("<0x0A>", "<|new_line|>")
                elif key == "logprob":
                    new_dict["prob"] = math.exp(value)
                elif key == "entropy":
                    new_dict["entropy"] = value
                else:
                    print(f"Unknown key: {key}, expected token_id, logprob, entropy")
            ls.append(new_dict)

        if ref_values_ls is not None:
            for i, (values, ref_values) in enumerate(zip(values_ls, ref_values_ls)):
                assert values["token_id"] == ref_values["token_id"]
                for key, value in values.items():
                    if key == "logprob":
                        ls[i]["ref_prob"] = math.exp(ref_values[key])
                        ls[i]["diff_prob"] = ls[i]["prob"] - ls[i]["ref_prob"]
                        ls[i]["diff_logprob"] = value - ref_values[key]
                    elif key == "entropy":
                        ls[i]["ref_entropy"] = ref_values[key]
                        ls[i]["diff_entropy"] = ls[i]["entropy"] - ls[i]["ref_entropy"]
        return ls

    def gen_and_cal_prob(self, text, max_new_tokens, temperature, use_beam_search, top_k, top_p, best_of, return_prompt):
        logs = "<b>Stats:</b><br>"
        input_text = [self.chat_template(text)]
        input_ids = self.tokenizer(input_text)["input_ids"]

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
        values_ls = self.get_logprob_and_entropy(self.llm, input_ids, output_ids)
        ref_values_ls = None

        if self.ref_model_name is not None:
            ref_values_ls = self.get_logprob_and_entropy(self.ref_llm, input_ids, output_ids)

        logs += f"cal logs time: {time.time() - start:.2f} s<br>"
        values = self.cal_stats(values_ls, ref_values_ls)

        logs += f"total logprob: {sum([math.log(v['prob']) for v in values]):.2f}, logprob/token: {sum([math.log(v['prob']) for v in values]) / len(values):.2f}<br>"
        if values[0].get("entropy") is not None:
            logs += f"total entropy: {sum([v['entropy'] for v in values]):.2f}, entropy/token: {sum([v['entropy'] for v in values]) / len(values):.2f}<br>"
        if values[0].get("ref_prob") is not None:
            logs += f"total ref logprob: {sum([math.log(v['ref_prob']) for v in values]):.2f}, ref logprob/token: {sum([math.log(v['ref_prob']) for v in values]) / len(values):.2f}<br>"
            if values[0].get("ref_entropy") is not None:
                logs += f"total ref entropy: {sum([v['ref_entropy'] for v in values]):.2f}, ref entropy/token: {sum([v['ref_entropy'] for v in values]) / len(values):.2f}<br>"

            logs += f"total diff logprob: {sum([v['diff_logprob'] for v in values]):.2f}, diff logprob/token: {sum([v['diff_logprob'] for v in values]) / len(values):.2f}<br>"
            logs += f"total diff entropy: {sum([v['diff_entropy'] for v in values]):.2f}, diff entropy/token: {sum([v['diff_entropy'] for v in values]) / len(values):.2f}<br>"
        logs += f"num tokens: {len(values)}<br>"
        logs += f"num tokens per second: {len(values) / generate_time:.2f}<br>"

        return logs, values

    def to_html_str(self, logs: str, values) -> [str, str]:
        """values is a list of dictionaries [{'text': 'Hello', 'prob': 0.5, 'entropy'}, ...]"""
        keys_map = {"diff_entropy": 1, "diff_logprob": 2, "diff_prob": 3, "entropy": 4, "ref_entropy": 5, "prob": 6, "ref_prob": 7}  # set the order
        keys_unorder = [key for key in values[0].keys() if key != "text"]
        keys = sorted(keys_unorder, key=lambda x: keys_map.get(x, 0))
        colored_text = [""] * len(keys)
        # add legends to colored_text
        for i, key in enumerate(keys):
            colored_text[i] += f"<br><strong>{key}:</strong><br>"
        log_info = logs + "<br><b>Legend:</b><br>"
        for item in values:
            text = item["text"] if item["text"] != "<|new_line|>" else " <br>"
            log_info += f"{text}: "
            for i, key in enumerate(keys):
                background_color = color_map(item[key], key)
                colored_text[i] += f"<span style='background-color: {background_color}; padding: 0px;'>{text}</span>"
                log_info += f"{key}: <span style='background-color: {background_color}; padding: 0px;'>{item[key]:.2f}</span>, "
            log_info += "<br>"
        return "<br>".join(colored_text), log_info

    def run(self):
        self.iface.launch(share=True)


@dataclass
class ScriptArguments:
    model_name: str = "openchat/openchat_3.5"
    use_flash_attention_2: bool = False
    ref_model_name: Optional[str] = None
    low_cpu_mem_usage: bool = True


args = tyro.cli(ScriptArguments)


# model_name = "openchat/openchat_3.5"
# model_name = "lvwerra/gpt2-imdb"
# model_name = "berkeley-nest/Starling-LM-7B-alpha"
extra_generate_kwargs = {}
server = VisualizeLLM(
    args.model_name, args.ref_model_name, extra_generate_kwargs, openchat_template, args.use_vllm, args.use_flash_attention_2, args.low_cpu_mem_usage
)
server.run()
