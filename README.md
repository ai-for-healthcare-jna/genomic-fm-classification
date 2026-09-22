# Genomic Foundation Model Fine-Tuning: Human Enhancer Classification

Linear-probe evaluation of [HyenaDNA](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf)'s pretrained representations on binary classification of DNA sequences as human enhancers or not, on the [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) `human_enhancers_cohn` task.

## Goal

The question this answers: how much task-relevant signal is already present in HyenaDNA's frozen, pretrained representations, before any task-specific adaptation? The backbone is frozen entirely and only a linear head is trained on top, which is the standard protocol for isolating representation quality from fine-tuning capacity. Published full-fine-tuning numbers are used below as context for reading the result — how much headroom adaptation adds on top of the frozen representation — not as a target this run was set up to match.

## Task

Enhancers are non-coding regulatory DNA regions that increase the activity of nearby genes. Given a ~500bp DNA sequence, the model predicts whether it functions as a human enhancer (binary classification, balanced 50/50 classes across 20,843 train / 6,948 test examples).

## Approach

**Linear probing, not full fine-tuning.** HyenaDNA is pretrained on the human reference genome with a next-nucleotide prediction objective. The backbone (436,096 parameters) is frozen with zero gradient updates; a 258-parameter linear layer on top is the only thing trained. A layer this small has little capacity to compensate for weak representations, so its performance is attributable to what pretraining learned, not to fine-tuning capacity.

**Architecture-driven design choices:**
- HyenaDNA is a causal, convolution/state-space architecture (not attention-based), which avoids the quadratic scaling of transformer attention with sequence length.
- Because it's causal (token *i* only ever sees tokens ≤ *i*), only the *last* token's hidden state has seen the full sequence — so the classifier pools that last non-padded token rather than mean-pooling across all positions.
- Tokenization is character-level (one nucleotide per token), not subword-based: DNA's 4-letter alphabet has none of the recurring, meaningful substructure that subword tokenization exploits in natural language, and grouping nucleotides into chunks would obscure single-nucleotide resolution that matters biologically.

## Pipeline

- `src/data_loading.py` — loads `katarinagresova/genomic_benchmarks_human_enhancers_cohn` from HuggingFace, reports class balance, and produces a stratified train/val split (test set stays untouched until final evaluation).
- `src/model.py` — frozen HyenaDNA backbone + linear classification head.
- `src/train.py` — trains only the head on the full train/val split, times the run, evaluates on the full test set at the end, and writes `results/classifier_head.pt` + `results/metrics.json`.
- `src/evaluate.py` — standalone re-evaluation of a saved head on the test set (accuracy + F1).

## Results

Trained on an AWS `g4dn.xlarge` (T4 GPU): 8 epochs, batch size 32, Adam lr 1e-3, 2m41s total.

| | Accuracy | F1 |
|---|---|---|
| CNN baseline, trained from scratch (Grešová et al., 2023) | 69.5% | 67.1% |
| HyenaDNA-tiny, full fine-tune, single run (Nguyen et al., 2023) | 74.2% | — |
| HyenaDNA-tiny, full fine-tune, 5-fold CV (Schiff et al., 2024 — Caduceus) | 72.9% ± 1.4% | — |
| **HyenaDNA-tiny, frozen backbone + linear probe (this run)** | **66.4%** | **68.2%** |

Train and validation metrics tracked each other closely through all 8 epochs (final: train_acc 66.4% / val_acc 65.7%, train_loss 0.604 / val_loss 0.605) and both plateaued by epoch 5-6 — no divergence between them, so no overfitting signature, consistent with a 258-parameter head having little room to memorize training-set noise.

The frozen-probe number sits below both full-fine-tuning numbers, which is expected: those update all 436K backbone parameters on this task, this run updates none of them. It also sits slightly below the from-scratch CNN baseline, which is the more interesting comparison — it says HyenaDNA's *frozen* pretrained representations alone, without adaptation, aren't yet sufficient to beat a small task-specific model trained from nothing on this task.

## Limitations & Open Questions

- **Single-split evaluation.** This result comes from one train/val/test split, not cross-validated — the full-fine-tune numbers above use 5-fold CV. Some of the observed gap could be split-variance rather than solely an effect of freezing the backbone. A matched-protocol comparison (same hyperparameters, same CV scheme, full fine-tuning run through this same codebase) would isolate that.
- **No invariance check.** Enhancer function is strand-agnostic biologically, but this linear probe has no architectural guarantee of recognizing a sequence and its reverse complement equally — unlike architectures such as Caduceus, built specifically for that equivariance. Whether HyenaDNA's frozen representations already encode that symmetry, or the probe is relying on strand-specific artifacts, is untested here.

## Implementation notes

HyenaDNA's tokenizer defaults to left-padding, but last-token pooling assumes right-padding (its index math finds the last real token by counting from the start of the sequence). Fixed by explicitly setting `tokenizer.padding_side = "right"` in `model.py`. This dataset's sequences are all fixed-length, so no batch actually triggers real padding — the bug was latent and had no effect on the results above, but would misfire on any batch with genuine length variance.

## Running

```bash
pip install -r requirements.txt

python3 src/data_loading.py   # sanity-check the dataset
python3 src/model.py          # shape/param smoke test
python3 src/train.py --max_train_samples 20 --max_val_samples 20 --max_test_samples 20   # fast CPU smoke test

python3 src/train.py --epochs 8 --batch_size 32 --device cuda   # full training run (GPU recommended)
python3 src/evaluate.py --device cuda                            # re-evaluate a saved checkpoint on the full test set
```
