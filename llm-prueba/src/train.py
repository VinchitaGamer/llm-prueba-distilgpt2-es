import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def read_text_dataset(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    docs = [d for d in txt.split("\n\n") if len(d.strip()) > 0]
    return docs


def chunk_tokenize(tokenizer, texts, block_size: int):
    tokenized = tokenizer(texts, return_attention_mask=False)
    input_ids = []
    for ids in tokenized["input_ids"]:
        input_ids.extend(ids + [tokenizer.eos_token_id])
    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    blocks = [input_ids[i : i + block_size] for i in range(0, total_length, block_size)]
    return Dataset.from_dict({"input_ids": blocks, "labels": blocks})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--dataset_path", default="data/dataset.txt")
    parser.add_argument("--checkpoint_dir", default="models/checkpoints")
    parser.add_argument("--final_model_dir", default="models/final_model")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.final_model_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = read_text_dataset(args.dataset_path)

    n_eval = max(1, int(len(texts) * args.eval_ratio)) if args.eval_ratio > 0 else 0
    eval_texts = texts[:n_eval] if n_eval > 0 else []
    train_texts = texts[n_eval:] if n_eval > 0 else texts

    train_ds = chunk_tokenize(tokenizer, train_texts, args.block_size)
    eval_ds = chunk_tokenize(tokenizer, eval_texts, args.block_size) if n_eval > 0 else None

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if model.config.pad_token_id is None and tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        fp16=fp16,
        report_to=["tensorboard"],
        logging_dir="logs",
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        eval_steps=args.save_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Auto-detect latest checkpoint if resuming
    resume_cp = None
    if args.resume and os.path.isdir(args.checkpoint_dir):
        candidates = [
            os.path.join(args.checkpoint_dir, d)
            for d in os.listdir(args.checkpoint_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.checkpoint_dir, d))
        ]
        if candidates:
            def step_key(p: str) -> int:
                name = os.path.basename(p)
                try:
                    return int(name.split("-")[-1])
                except Exception:
                    return -1
            candidates.sort(key=step_key, reverse=True)
            resume_cp = candidates[0]
            print(f"Resuming from checkpoint: {resume_cp}")
        else:
            print("No checkpoint found to resume; starting fresh.")

    train_result = trainer.train(resume_from_checkpoint=resume_cp)

    try:
        metrics = {}
        if eval_ds is not None:
            metrics = trainer.evaluate()
        metrics.update(train_result.metrics)
        with open(Path("logs") / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False))
    except Exception as e:
        print(str(e))

    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)

    print(f"Saved final model to {args.final_model_dir}")


if __name__ == "__main__":
    main()
