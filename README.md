# IndoBART IndoSum Summarizer

An abstractive text summarization project for Indonesian news articles using IndoBART, trained on the IndoSum dataset.

This project focuses not only on model performance, but also on handling real-world engineering issues such as dataset integrity and legacy tokenizer compatibility within the modern Hugging Face ecosystem.

---

## Overview

* **Task**: Abstractive Text Summarization (Indonesian)
* **Model**: `indobenchmark/indobart-v2`
* **Dataset**: IndoSum
* **Framework**: Hugging Face Transformers
* **Environment**: Kaggle (GPU T4)

---

## Key Highlights

### 1. Data Integrity Audit

A significant portion of the IndoSum dataset contains duplicate entries.

* ~72% of the data (≈51,994 rows) were duplicates
* Final training set: **19,359 unique samples**

This project prioritizes **data quality over quantity**, which has a direct impact on model generalization.

---

### 2. Tokenizer Compatibility (Monkey Patching)

The original `IndoNLGTokenizer` is not fully compatible with the latest Hugging Face `transformers` API.

To ensure training stability, a lightweight compatibility layer was implemented via monkey patching:

* **Padding fix**: ignores unsupported `padding_side` argument
* **Vocabulary stub**: bypasses incomplete `save_vocabulary` implementation
* **Decode normalization**: aligns method signature with updated API

This approach avoids modifying the original library while maintaining reproducibility.

---

## Training Setup

* **Max Input Length**: 1024
* **Max Target Length**: 128
* **Decoding Strategy**: Beam Search (num_beams=4)
* **Length Penalty**: 2.0
* **Early Stopping**: patience = 4

---

## Results (Epoch 3)

* **ROUGE-1**: 0.3540
* **ROUGE-2**: 0.3108
* **ROUGE-L**: 0.3447
* **Validation Loss**: 0.5067

---

## Repository Structure

```
src/                # core pipeline (training, preprocessing, evaluation)
notebooks/          # experimentation and exploration
configs/            # training configuration
results/            # evaluation outputs
```

---

## Notes

* The monkey patching layer is a **temporary compatibility solution** and may break with future library updates.
* This project is designed for reproducibility within constrained environments (e.g., Kaggle GPU T4).

---

## Future Work

* Baseline comparison (e.g., extractive methods)
* Model optimization and hyperparameter tuning
* Deployment as API or demo application

---
