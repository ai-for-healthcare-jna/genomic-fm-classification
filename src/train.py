"""
Train the linear head on the frozen HyenaDNA backbone (see model.py), then
evaluate on the full test set and save metrics for the writeup.

Defaults (epochs=8, batch_size=32) target a single T4; override both from
the CLI. `--max_train_samples`/`--max_val_samples`/`--max_test_samples`
still exist for CPU smoke tests but default to None (full dataset).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loading import get_tiny_subset, load_enhancer_dataset, split_train_val
from evaluate import evaluate
from model import HyenaDNAClassifier, load_tokenizer, make_collate_fn

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_epoch(model, loader, optimizer, device, train: bool):
    """Shared logic for one pass over a DataLoader, in train or eval mode."""
    model.classifier.train(train)

    total_loss, total_correct, total_examples = 0.0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        with torch.set_grad_enabled(train):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m{secs:04.1f}s"


def train(args: argparse.Namespace) -> HyenaDNAClassifier:
    device = torch.device(args.device)

    tokenizer = load_tokenizer()
    model = HyenaDNAClassifier().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    dataset = load_enhancer_dataset()
    train_val = split_train_val(dataset["train"])
    train_split, val_split = train_val["train"], train_val["val"]
    test_split = dataset["test"]

    if args.max_train_samples:
        train_split = get_tiny_subset(train_split, args.max_train_samples)
    if args.max_val_samples:
        val_split = get_tiny_subset(val_split, args.max_val_samples)
    if args.max_test_samples:
        test_split = get_tiny_subset(test_split, args.max_test_samples)

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_split, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"training on {len(train_split)} examples, validating on {len(val_split)}")

    training_start = time.perf_counter()
    train_loss = train_acc = val_loss = val_acc = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        epoch_time = time.perf_counter() - epoch_start

        # Overfitting signature to watch for: train_loss keeps dropping /
        # train_acc keeps climbing while val_loss stops improving or rises
        # (val_acc plateaus or drops) -- the head is fitting noise specific
        # to the training sample rather than a generalizable signal.
        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"epoch_time={epoch_time:.1f}s"
        )

        if epoch == 1:
            remaining_epochs = args.epochs - 1
            estimated_remaining = epoch_time * remaining_epochs
            estimated_total = epoch_time * args.epochs
            print(
                f"  [timing] first epoch took {epoch_time:.1f}s -> "
                f"estimated remaining: {format_seconds(estimated_remaining)}, "
                f"estimated total: {format_seconds(estimated_total)}"
            )

    total_training_time = time.perf_counter() - training_start
    print(f"\ntotal training time: {format_seconds(total_training_time)}")

    test_accuracy, test_f1 = evaluate(model, test_loader, device)
    print(f"\n=== Test set results ({len(test_split)} examples) ===")
    print(f"accuracy: {test_accuracy:.4f}")
    print(f"f1:       {test_f1:.4f}")
    print("Compare against the published Genomic Benchmarks baseline for human_enhancers_cohn.")

    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_path = RESULTS_DIR / "classifier_head.pt"
    torch.save(model.classifier.state_dict(), checkpoint_path)
    print(f"\nsaved trained head to {checkpoint_path}")

    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "total_training_time_seconds": total_training_time,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved metrics to {metrics_path}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear head on frozen HyenaDNA")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training set size (e.g. 20 for a fast smoke test). Default: full split.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Cap validation set size (e.g. 20 for a fast smoke test). Default: full split.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Cap test set size (e.g. 20 for a fast smoke test). Default: full split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
