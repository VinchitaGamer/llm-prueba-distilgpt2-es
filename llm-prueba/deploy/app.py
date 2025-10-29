import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_pipeline():
    model_id = os.environ.get("MODEL_ID") or os.environ.get("LOCAL_MODEL_DIR") or "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
    return pipe, tok

pipe, tokenizer = load_pipeline()

def generate(prompt, max_new_tokens, temperature, top_p, repetition_penalty, seed):
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = pipe(
        prompt,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        generator=gen,
    )[0]["generated_text"]
    return out

with gr.Blocks() as demo:
    gr.Markdown("# LLM Prueba – Demo")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=6, value="Escribe una breve historia sobre un robot y un gato")
            max_new_tokens = gr.Slider(16, 256, value=128, step=1, label="max_new_tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top_p")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="repetition_penalty")
            seed = gr.Number(value=42, label="seed")
            btn = gr.Button("Generar")
        with gr.Column():
            out = gr.Textbox(label="Salida", lines=20)
    btn.click(generate, [prompt, max_new_tokens, temperature, top_p, repetition_penalty, seed], out)

if __name__ == "__main__":
    demo.launch()
