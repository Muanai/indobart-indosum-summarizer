# IndoBART-v2 Abstractive Summarization Pipeline 🇮🇩

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-latest-green.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Read on Medium](https://img.shields.io/badge/Read_the_Article-Medium-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@muanaikhalifahr/taming-ambiguity-in-indonesian-text-summarization-building-an-indobart-v2-pipeline-from-research-f3be68ee6cac)

> **Read the full technical breakdown on [Medium](https://medium.com/@muanaikhalifahr/taming-ambiguity-in-indonesian-text-summarization-building-an-indobart-v2-pipeline-from-research-f3be68ee6cac):** *Beyond the ROUGE Score: A Practical Engineering Guide to Indonesian NLP Pipelines.*

An engineering-focused, production-ready abstractive text summarization pipeline utilizing **IndoBART-v2** fine-tuned on the **IndoSum** dataset. This project goes beyond basic model training to address real-world machine learning engineering challenges, including legacy dependency resolution, hardware-specific optimization, and modular system design.

Designed with high-density information environments in mind, such as the Indonesian Fintech and E-commerce sectors.

## Engineering Highlights

This repository is structured around scalable software engineering principles rather than monolithic notebook scripts:

*   **Modular Architecture:** The pipeline is strictly divided into functional classes (`IndoSumManager`, `DataPreprocessor`, `SummarizationTrainer`) to ensure separation of concerns, making the codebase highly maintainable and scalable.
*   **Legacy Dependency Monkey-Patching (`patcher.py`):** Includes a runtime surgical patch to resolve critical signature mismatches between the aging `indobenchmark` tokenizer toolkit and modern Hugging Face `transformers` APIs, allowing legacy Indonesian tokenizers to run in modern environments without source-code modification.
*   **Hardware-Aligned Training:** Optimized for NVIDIA Turing architectures (e.g., T4) utilizing Tensor Cores. Implements Mixed Precision (FP16) training to maximize throughput and stability, proving that architectural alignment often outweighs raw VRAM capacity.
*   **Skeptical Evaluation:** Emphasizes realistic generation parameters (e.g., controlling the End-of-Sequence looping) and monitors the training/validation loss divergence to prevent the "overfitting trap," rather than relying solely on ROUGE scores.

## Project Structure

```text
├── data/                  # Raw and processed IndoSum datasets (Not tracked via Git)
├── src/
│   ├── data_manager.py    # IndoSumManager: Data extraction, cleaning, and flattening
│   ├── preprocessor.py    # DataPreprocessor: Tokenization, padding, and deterministic splitting
│   ├── trainer.py         # SummarizationTrainer: Wraps HF Trainer, DataCollator, and Callbacks
│   └── patcher.py         # apply_tokenizer_patches: Runtime
```

## Installation
Clone the repository:

```bash
git clone [https://github.com/muanaikhalifah/indobart-summarization.git](https://github.com/muanaikhalifah/indobart-summarization.git)
cd indobart-summarization
```
Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Patching the Tokenizer
Before initializing the training loop, ensure the legacy tokenizer is patched to prevent AttributeError and method signature conflicts:

```python
from transformers import AutoTokenizer
from src.patcher import apply_tokenizer_patches

# Load legacy tokenizer
raw_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobart-v2")

# Apply runtime surgery
tokenizer = apply_tokenizer_patches(raw_tokenizer)
```
2. Precise Inference Generation
To generate summaries without encountering End-of-Sequence (EOS) repetition or noise, use strict generation parameters:

```Python
def generate_summary(text, model, tokenizer, device):
    id_id_token = tokenizer.convert_tokens_to_ids("[ind]")
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=128,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
        forced_bos_token_id=id_id_token,
        decoder_start_token_id=id_id_token
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```
## Evaluation & Metrics
While the model achieves competitive ROUGE scores (ROUGE-1: ~0.35), the primary evaluation focus is on semantic groundedness and loss divergence.

Training logs indicate that validation loss begins to diverge around Epoch 3, signaling a shift from generalized learning to memorization. For production deployment, it is highly recommended to utilize the weights saved at the divergence point (Early Stopping) rather than the final epoch to maintain factual consistency and reduce hallucinations.

## License
This project is licensed under the MIT License - see the LICENSE file for details.