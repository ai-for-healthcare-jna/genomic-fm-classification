"""
Train the linear head on the frozen HyenaDNA backbone (see model.py).

Defaults sized for a single T4; `--max_train_samples`/`--max_val_samples`
cap dataset size for CPU smoke tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loading import get_tiny_subset, load_enhancer_dataset, split_train_val
from model import HyenaDNAClassifier, build_attention_mask, load_tokenizer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_collate_fn(tokenizer):
    """Tokenize a batch on the fly, padded to the batch's own max length."""

    def collate(batch):
        seqs = [ex["seq"] for ex in batch]
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        input_ids = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        attention_mask = build_attention_mask(input_ids, tokenizer)
        return input_ids, attention_mask, labels

    return collate


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


def train(args: argparse.Namespace) -> HyenaDNAClassifier:
    device = torch.device(args.device)

    tokenizer = load_tokenizer()
    model = HyenaDNAClassifier().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    dataset = load_enhancer_dataset()
    train_val = split_train_val(dataset["train"])
    train_split, val_split = train_val["train"], train_val["val"]

    if args.max_train_samples:
        train_split = get_tiny_subset(train_split, args.max_train_samples)
    if args.max_val_samples:
        val_split = get_tiny_subset(val_split, args.max_val_samples)

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"training on {len(train_split)} examples, validating on {len(val_split)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_path = RESULTS_DIR / "classifier_head.pt"
    torch.save(model.classifier.state_dict(), checkpoint_path)
    print(f"saved trained head to {checkpoint_path}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear head on frozen HyenaDNA")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training set size (e.g. 20 for a fast smoke test).",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Cap validation set size (e.g. 20 for a fast smoke test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
