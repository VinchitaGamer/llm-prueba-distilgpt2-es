import argparse
import os
import re
from pathlib import Path
from glob import glob

from datasets import load_dataset


def clean_text(s: str, lowercase: bool = False) -> str:
    s = s.replace("\r", " ")
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if lowercase:
        s = s.lower()
    return s


def load_local_texts(raw_dir: str) -> list[str]:
    paths = sorted(glob(os.path.join(raw_dir, "*.txt")))
    texts = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except UnicodeDecodeError:
            with open(p, "r", encoding="latin-1") as f:
                texts.append(f.read())
    return texts


def load_hf_opus_books_es(max_examples: int) -> list[str]:
    ds = load_dataset("opus_books", "es-en", split="train")
    texts = []
    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        t = ex.get("translation", {}).get("es")
        if t:
            texts.append(t)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf_opus_books", "local"], default="hf_opus_books")
    parser.add_argument("--output", default="data/dataset.txt")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--max_examples", type=int, default=50000)
    parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    if args.source == "local":
        texts = load_local_texts(args.raw_dir)
    else:
        texts = load_hf_opus_books_es(args.max_examples)

    cleaned = [clean_text(t, lowercase=args.lowercase) for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    joined = "\n\n".join(cleaned)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(joined)

    print(f"Wrote {len(cleaned)} documents -> {args.output}")


if __name__ == "__main__":
    main()
