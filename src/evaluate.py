"""
Evaluate a trained classification head on the test split.

Reports accuracy and F1 -- F1 is the metric the Genomic Benchmarks paper
reports for this task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from data_loading import get_tiny_subset, load_enhancer_dataset
from model import HyenaDNAClassifier, load_tokenizer
from train import make_collate_fn

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@torch.no_grad()
def evaluate(model: HyenaDNAClassifier, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.classifier.eval()
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return accuracy, f1


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    tokenizer = load_tokenizer()
    model = HyenaDNAClassifier().to(device)

    checkpoint_path = RESULTS_DIR / "classifier_head.pt"
    model.classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_split = load_enhancer_dataset()["test"]
    if args.max_test_samples:
        test_split = get_tiny_subset(test_split, args.max_test_samples)

    loader = DataLoader(
        test_split, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer)
    )

    accuracy, f1 = evaluate(model, loader, device)

    print(f"\n=== Test set results ({len(test_split)} examples) ===")
    print(f"accuracy: {accuracy:.4f}")
    print(f"f1:       {f1:.4f}")
    print("\nCompare against the published Genomic Benchmarks baseline for human_enhancers_cohn.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained head on the test split")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Cap test set size (e.g. 20 for a fast smoke test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
