# Genomic Foundation Model Fine-Tuning: Human Enhancer Classification

Linear-probe fine-tuning of [HyenaDNA](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf) for binary classification of DNA sequences as human enhancers or not, benchmarked against the published [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) baseline for the `human_enhancers_cohn` task.

## Task

Enhancers are non-coding regulatory DNA regions that increase the activity of nearby genes. Given a ~500bp DNA sequence, the model predicts whether it functions as a human enhancer (binary classification, balanced 50/50 classes).

## Approach

**Linear probing, not full fine-tuning.** HyenaDNA is pretrained on the human reference genome with a next-nucleotide prediction objective. To evaluate the quality of those pretrained representations in isolation, the backbone is frozen entirely (zero gradient updates) and only a new linear layer on top is trained. A linear layer has minimal capacity to compensate for weak representations, so its performance is attributable to the pretrained features rather than to fine-tuning capacity — this is the standard evaluation protocol for benchmarking foundation model representations.

**Architecture-driven design choices:**
- HyenaDNA is a causal, convolution/state-space architecture (not attention-based), which avoids the quadratic scaling of transformer attention with sequence length.
- Because it's causal (token *i* only ever sees tokens ≤ *i*), only the *last* token's hidden state has seen the full sequence — so the classifier pools that last non-padded token rather than mean-pooling across all positions.
- Tokenization is character-level (one nucleotide per token), not subword-based: DNA's 4-letter alphabet has none of the recurring, meaningful substructure that subword tokenization exploits in natural language, and grouping nucleotides into chunks would obscure single-nucleotide resolution that matters biologically.

## Pipeline

- `src/data_loading.py` — loads `katarinagresova/genomic_benchmarks_human_enhancers_cohn` from HuggingFace, reports class balance, and produces a stratified train/val split (test set stays untouched until final evaluation).
- `src/model.py` — frozen HyenaDNA backbone + linear classification head.
- `src/train.py` — trains only the head; the optimizer is scoped to the head's parameters specifically, on top of the backbone already being frozen.
- `src/evaluate.py` — reports accuracy and F1 on the test set, F1 to match the published benchmark's reporting protocol.

## Results

_Pending: to be filled in after the full training run (currently only pipeline-verified on a tiny CPU smoke test)._

| | Accuracy | F1 |
|---|---|---|
| This linear probe | — | — |
| Published Genomic Benchmarks baseline | — | — |

## Notable finding

HyenaDNA's tokenizer defaults to left-padding, but last-token pooling assumes right-padding (its index math finds the last token by counting real tokens from the start of the sequence). Fixed by explicitly setting `tokenizer.padding_side = "right"`. This dataset's sequences are all fixed-length, so no batch actually triggers real padding — meaning the bug was latent and had no effect on results here, but would misfire on any batch with genuine length variance.

## Running

```bash
pip install -r requirements.txt

python3 src/data_loading.py   # sanity-check the dataset
python3 src/model.py          # shape/param smoke test
python3 src/train.py --max_train_samples 20 --max_val_samples 20   # fast CPU smoke test
python3 src/evaluate.py --max_test_samples 20

python3 src/train.py          # full training run (GPU recommended)
python3 src/evaluate.py
```
