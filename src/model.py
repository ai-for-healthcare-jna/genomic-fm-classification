"""
Frozen HyenaDNA backbone + linear classification head.

Linear probe, not fine-tuning: backbone frozen, only `classifier` trains.
We use `AutoModel` rather than `AutoModelForSequenceClassification` so the
pooling below is explicit rather than buried in the HF port's own head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

BACKBONE_CHECKPOINT = "LongSafari/hyenadna-tiny-1k-seqlen-hf"


def load_tokenizer(checkpoint: str = BACKBONE_CHECKPOINT) -> PreTrainedTokenizerBase:
    """Load HyenaDNA's tokenizer (character-level, custom class -> trust_remote_code)."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    # Ships with a [PAD] token already; fallback here in case a future
    # checkpoint doesn't.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force right-padding: _last_token_pool assumes it, but this tokenizer
    # defaults to left-padding.
    tokenizer.padding_side = "right"

    return tokenizer


def load_frozen_backbone(checkpoint: str = BACKBONE_CHECKPOINT) -> PreTrainedModel:
    """Load the pretrained HyenaDNA backbone with all parameters frozen."""
    backbone = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()  # keep any dropout/batchnorm layers in inference mode

    return backbone


def build_attention_mask(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """HyenaDNA's tokenizer doesn't return attention_mask, so derive it from padding."""
    return (input_ids != tokenizer.pad_token_id).long()


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool via each sequence's last non-padded hidden state.

    Causal model: only the last real token has seen the full sequence, so
    mean-pooling would mix in under-context representations.
    """
    last_token_idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, last_token_idx]


class HyenaDNAClassifier(nn.Module):
    """Frozen HyenaDNA backbone + a linear head for binary sequence classification."""

    def __init__(self, checkpoint: str = BACKBONE_CHECKPOINT, num_labels: int = 2):
        super().__init__()
        self.backbone = load_frozen_backbone(checkpoint)
        hidden_size = self.backbone.config.d_model  # Hyena's naming for hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Hyena backbone has no attention_mask param (it's conv/SSM-based,
        # not attention); the mask is only used below for pooling.
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids)
        pooled = _last_token_pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)

    def trainable_parameters(self):
        return self.classifier.parameters()


if __name__ == "__main__":
    # Shape/smoke check.
    tokenizer = load_tokenizer()
    model = HyenaDNAClassifier()

    sample_seqs = ["ACTGACTGACTG", "GGGGCCCCAAAA"]
    input_ids = tokenizer(sample_seqs, padding=True, return_tensors="pt")["input_ids"]
    attention_mask = build_attention_mask(input_ids, tokenizer)
    logits = model(input_ids, attention_mask)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_frozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"trainable params (head): {n_trainable}")
    print(f"frozen params (backbone): {n_frozen}")
