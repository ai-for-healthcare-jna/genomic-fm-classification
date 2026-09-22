"""
Data loading for the Genomic Benchmarks "human_enhancers_cohn" task
(binary: enhancer vs. not, 500bp sequences). Train/test splits are fixed
by the HF mirror; we only carve train/val ourselves, for model selection.
"""

from __future__ import annotations

from collections import Counter

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

HF_DATASET_ID = "katarinagresova/genomic_benchmarks_human_enhancers_cohn"


def load_enhancer_dataset() -> DatasetDict:
    """Returns a DatasetDict with "train"/"test" splits ("seq": str, "label": 0/1)."""
    return load_dataset(HF_DATASET_ID)


def split_train_val(train_split: Dataset, val_fraction: float = 0.1, seed: int = 42) -> DatasetDict:
    """Stratified train/val split (fixed seed for reproducibility)."""
    # stratify_by_column needs a ClassLabel; the HF mirror stores label as
    # a plain int.
    train_split = train_split.cast_column("label", ClassLabel(names=["0", "1"]))
    split = train_split.train_test_split(test_size=val_fraction, seed=seed, stratify_by_column="label")
    return DatasetDict(train=split["train"], val=split["test"])


def summarize_split(dataset: Dataset, name: str, n_examples: int = 3) -> None:
    """Print class balance and a few (sequence, label) pairs for sanity checking."""
    labels = dataset["label"]
    counts = Counter(labels)
    total = len(labels)
    print(f"\n=== {name} split ===")
    print(f"total examples: {total}")
    for label, count in sorted(counts.items()):
        pct = 100 * count / total
        print(f"  label={label}: {count} ({pct:.1f}%)")

    print(f"example sequences (first {n_examples}):")
    for i in range(min(n_examples, total)):
        seq = dataset[i]["seq"]
        label = dataset[i]["label"]
        preview = seq if len(seq) <= 60 else f"{seq[:60]}..."
        print(f"  [label={label}] len={len(seq)} seq={preview}")


def get_tiny_subset(dataset: Dataset, n: int = 20, seed: int = 42) -> Dataset:
    """Shuffled slice of a split, for CPU smoke tests before spending GPU time."""
    return dataset.shuffle(seed=seed).select(range(min(n, len(dataset))))


if __name__ == "__main__":
    ds = load_enhancer_dataset()
    summarize_split(ds["train"], "train")
    summarize_split(ds["test"], "test")

    train_val = split_train_val(ds["train"])
    summarize_split(train_val["train"], "train (post train/val split)")
    summarize_split(train_val["val"], "val")
