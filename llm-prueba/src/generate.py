import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/final_model")
    parser.add_argument("--output", default="outputs/examples.txt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=None,
    )

    prompts = [
        "Resumen de la noticia:",
        "Escribe una receta sencilla para preparar",
        "Había una vez en una ciudad pequeña",
    ]

    outputs = []
    for p in prompts:
        out = pipe(
            p,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            generator=gen,
        )[0]["generated_text"]
        outputs.append((p, out))

    with open(args.output, "w", encoding="utf-8") as f:
        for p, out in outputs:
            f.write("PROMPT:\n")
            f.write(p + "\n\n")
            f.write("OUTPUT:\n")
            f.write(out + "\n\n" + ("-" * 80) + "\n\n")

    print(f"Wrote generations to {args.output}")


if __name__ == "__main__":
    main()
