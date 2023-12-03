import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gradio as gr
from utils import logprobs_from_logits, color_map, openchat_template


class VisualizeLLM:
    def __init__(self, model_name, extra_generate_kwargs, chat_template) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.extra_generate_kwargs = extra_generate_kwargs
        if self.extra_generate_kwargs.get("pad_token_id") is None:
            self.extra_generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat_template = chat_template

        max_new_token_slider = gr.Slider(minimum=1, maximum=4096, step=1, label="Max New Tokens", value=256)
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Temperature", value=0.7)
        do_sample_checkbox = gr.Checkbox(label="Do Sample", value=True)
        top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Top K", value=0)
        top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, label="Top P", value=1)

        # custom_css = """
        #     .output_interface {
        #         width: 600px; /* Set the width of the output boxes */
        #         height: 300px; /* Set the height of the output boxes */
        #     }
        #     """

        self.iface = gr.Interface(
            fn=lambda *args: self.to_html_str(self.gen_and_cal_prob(*args)),
            inputs=["text", max_new_token_slider, temperature_slider, do_sample_checkbox, top_k_slider, top_p_slider],
            outputs=["html", "html"],
            # css=custom_css,
            theme="default",
            title="Visualize LLMs",
            description=f"The color visualize the log-probabilities of the generated tokens, from red (low probability) to green (high probability). Should expect the model to generate low probability if it feels uncertain. The model used is {model_name}.",
        )

    def prepare_input(self, text) -> dict:
        input_tensor = self.tokenizer(text)
        for key, val in input_tensor.items():
            input_tensor[key] = torch.tensor(val, dtype=torch.long)
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

    def create_prob_ls(self, stats):
        """create a list of dictionaries like [{'text': 'Hello', 'value': 0.5}, ...]"""
        ls = []
        for token, logprob in zip(stats["output_tokens"][0], stats["logprobs"][0]):
            ls.append({"text": self.tokenizer.convert_ids_to_tokens(token.item()).replace("▁", " "), "value": torch.exp(logprob).item()})
        return ls

    def gen_and_cal_prob(self, text, max_new_tokens, temperature, do_sample, top_k, top_p):
        input_text = [self.chat_template(text)]
        input_tensor = self.prepare_input(input_text)
        output_ids = self.model.generate(
            **input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            **self.extra_generate_kwargs,
        )
        stats = self.cal_logprob(input_tensor["input_ids"], output_ids)
        return self.create_prob_ls(stats)

    def to_html_str(self, data) -> [str, str]:
        """data is a list of dictionaries like [{'text': 'Hello', 'value': 0.5}, ...]"""
        colored_text = ""
        legend_info = "<br><br><b>Legend:</b><br>"
        for item in data:
            background_color = color_map(item["value"])
            colored_text += f"<span style='background-color: {background_color}; padding: 2px;'>{item['text']}</span>"
            legend_info += f"{item['text']}: Probability: {item['value']:.2f}, Color: {background_color}<br>"
        return colored_text, legend_info

    def run(self):
        self.iface.launch(share=True)


if __name__ == "__main__":
    model_name = "openchat/openchat_3.5"
    generate_kwargs = {}
    server = VisualizeLLM(model_name, generate_kwargs, openchat_template)
    server.run()
