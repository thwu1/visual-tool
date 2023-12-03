import math
import time

import gradio as gr
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import color_map, logprobs_from_logits, openchat_template


class VisualizeLLM:
    def __init__(self, model_name, extra_generate_kwargs, chat_template) -> None:
        self.llm = LLM(model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.extra_generate_kwargs = extra_generate_kwargs  # deprecated
        if self.extra_generate_kwargs.get("pad_token_id") is None:
            self.extra_generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat_template = chat_template
        max_new_token_slider = gr.Slider(minimum=1, maximum=8192, step=1, label="Max New Tokens", value=1024)
        temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Temperature", value=0.7)
        use_beam_search_box = gr.Checkbox(label="Use beam search", value=False)
        top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Top K", value=0)
        top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, label="Top P", value=1)
        best_of_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Best Of, for beam search", value=3)

        self.iface = gr.Interface(
            fn=lambda *args: self.to_html_str(*self.gen_and_cal_prob(*args)),
            inputs=["text", max_new_token_slider, temperature_slider, use_beam_search_box, top_k_slider, top_p_slider, best_of_slider],
            outputs=["html", "html"],
            theme="default",
            title="Visualize LLMs",
            description=f"The color visualize the log-probabilities of the generated tokens, from red (low probability) to green (high probability). Should expect the model to generate low probability if it feels uncertain. The model used is {model_name}.",
        )

    def prepare_input(self, text) -> dict:
        input_tensor = self.tokenizer(text)
        return input_tensor

    def cal_logprob(self, input_ids, output_ids):
        """
        input_ids: [batch_size, seq_len]
        output_ids: [batch_size, seq_len]"""
        num_gen_tokens = output_ids.shape[-1] - input_ids.shape[-1]
        # print(num_gen_tokens)
        assert torch.eq(output_ids[0][: input_ids.shape[-1]], input_ids[0]).all()
        input_kwargs = {
            "input_ids": output_ids,
            "attention_mask": torch.ones_like(output_ids),
            "return_dict": True,
        }
        logits = self.model(**input_kwargs).logits
        assert logits.shape[1] == input_ids.shape[-1] + num_gen_tokens
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_kwargs["input_ids"][:, 1:])
        return {"output_tokens": output_ids[:, -num_gen_tokens:], "logprobs": logprobs[:, -num_gen_tokens:]}

    def cal_prob(self, logprobs):
        """logprobs = [{token_id: logprob}, ...}]"""
        ls = []
        for logprob in logprobs:
            for token_id, logprob in logprob.items():
                ls.append({"text": self.tokenizer.convert_ids_to_tokens(token_id).replace("▁", " ").replace("<0x0A>","<|new_line|>"), "value": math.exp(logprob)})
        return ls

    def gen_and_cal_prob(self, text, max_new_tokens, temperature, use_beam_search, top_k, top_p, best_of):
        time_stats = "<b>Stats:</b><br>"
        input_text = [self.chat_template(text)]
        input_ids = self.tokenizer(input_text)["input_ids"]

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
        time_stats += f"generate time: {time.time() - start:.2f} s<br>"

        return self.cal_prob(outputs.logprobs), time_stats

    def to_html_str(self, data, time_stats: str) -> [str, str]:
        """data is a list of dictionaries like [{'text': 'Hello', 'value': 0.5}, ...]"""
        colored_text = ""
        legend_info = time_stats + "<br><b>Legend:</b><br>"
        for item in data:
            background_color = color_map(item["value"])
            text = item["text"] if item["text"] != "<|new_line|>" else " <br>"
            colored_text += f"<span style='background-color: {background_color}; padding: 0px;'>{text}</span>"
            legend_info += f"{item['text']}: Probability: {item['value']:.2f}, Color: {background_color}<br>"
        return colored_text, legend_info

    def run(self):
        self.iface.launch(share=True)


if __name__ == "__main__":
    model_name = "openchat/openchat_3.5"
    # model_name = "lvwerra/gpt2-imdb"
    extra_generate_kwargs = {}
    server = VisualizeLLM(model_name, extra_generate_kwargs, openchat_template)
    server.run()
