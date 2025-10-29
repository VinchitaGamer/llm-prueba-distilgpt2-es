import argparse
import os
import random

from datasets import load_dataset

def load_hf_opus_books_es(max_examples=None):
    """Loads Spanish texts from the Hugging Face opus_books dataset (en-es config)."""
    print(f"Loading opus_books dataset (en-es config) from Hugging Face...")
    # Use 'en-es' as the 'es-en' config is not available and extract 'es'
    ds = load_dataset("opus_books", "en-es", split="train")
    print(f"Loaded {len(ds)} examples.")

    texts = []
    # The dataset has pairs of 'en' and 'es' texts. We want the Spanish one.
    for example in ds:
        # Ensure Spanish text is present and is a string
        if example and isinstance(example.get('es'), str):
            texts.append(example['es'])


    if max_examples is not None and max_examples < len(texts):
        print(f"Selecting {max_examples} random examples out of {len(texts)}.")
        texts = random.sample(texts, max_examples)

    return texts

def load_local_texts(directory="data/raw", max_examples=None):
    """Loads texts from local .txt files."""
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            print(f"Loading {filepath}...")
            with open(filepath, "r", encoding="utf-8") as f:
                texts.extend(f.readlines())

    # Remove leading/trailing whitespace and filter out empty lines
    texts = [text.strip() for text in texts if text.strip()]

    if max_examples is not None and max_examples < len(texts):
        print(f"Selecting {max_examples} random examples out of {len(texts)}.")
        texts = random.sample(texts, max_examples)

    return texts

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for LLM training.")
    parser.add_argument("--source", type=str, default="hf_opus_books", choices=["hf_opus_books", "local"],
                        help="Source of the dataset: 'hf_opus_books' or 'local'.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to use. None for all.")
    parser.add_argument("--output", type=str, default="data/dataset.txt",
                        help="Path to save the prepared dataset.")

    args = parser.parse_args()

    if args.source == "hf_opus_books":
        texts = load_hf_opus_books_es(args.max_examples)
    elif args.source == "local":
        texts = load_local_texts(max_examples=args.max_examples)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    print(f"Saving {len(texts)} processed examples to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

if __name__ == "__main__":
    main()