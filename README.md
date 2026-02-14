# AutoAWQ-Reborn

<p align="center">
    <em>The actively maintained successor to AutoAWQ — 4-bit AWQ quantization for modern LLMs.</em>
</p>

<p align="center">
    <a href="https://github.com/BBC-Esq/AutoAWQ-Reborn/releases">
        <img alt="GitHub - Releases" src="https://img.shields.io/github/v/release/BBC-Esq/AutoAWQ-Reborn?style=flat-square&color=teal">
    </a>
    <a href="https://github.com/BBC-Esq/AutoAWQ-Reborn/blob/main/LICENSE">
        <img alt="License - MIT" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square">
    </a>
    <a href="https://huggingface.co/models?search=awq">
        <img alt="Hugging Face - Models" src="https://img.shields.io/badge/🤗_7000+_AWQ_models-8A2BE2?style=flat-square">
    </a>
    <a href="https://github.com/BBC-Esq/AutoAWQ-Reborn">
        <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square">
    </a>
</p>

---

## About

**AutoAWQ-Reborn** is a continuation of the [original AutoAWQ project](https://github.com/casper-hansen/AutoAWQ) by Casper Hansen, which pioneered accessible 4-bit quantization for large language models using the AWQ (Activation-aware Weight Quantization) algorithm. The original project was officially deprecated after an extraordinary run — over 2 million downloads, 7,000+ quantized models on Hugging Face, and 2.1k GitHub stars.

This fork picks up where the original left off, with a focus on:

- **Ongoing compatibility** with the latest versions of Transformers, PyTorch, and the broader Hugging Face ecosystem
- **Support for new model architectures** as they are released (Qwen 3, DeepSeek V3, Gemma 2, and more)
- **Modern Python support** (3.10 through 3.13)
- **Active maintenance** and community-driven development

We are grateful to Casper Hansen and the original contributors for building the foundation that AutoAWQ-Reborn continues to build upon.

## Key Features

- **3x faster inference** and **3x lower memory** compared to FP16 through 4-bit quantization
- **15 supported model architectures** including text and vision-language models
- **7 quantization backends** — GEMM, GEMV, GEMV Fast, ExLlama, ExLlama V2, Marlin, and IPEX
- **Fused modules** for accelerated inference with custom attention, normalization, and MoE fusion
- **Built-in evaluation** — perplexity, MMLU, HumanEval, LibriSpeech, and KL divergence
- **Multi-GPU** inference support
- **CPU inference** via Intel Extension for PyTorch (IPEX)
- **Full compatibility** with Hugging Face Transformers and vLLM

## Supported Models

| Architecture | Variants | Vision |
|:---|:---|:---:|
| LLaMA | LLaMA 2, LLaMA 3, LLaMA 3.1, Code Llama | |
| Mistral | Mistral 7B, v0.2, v0.3 | |
| Mixtral | Mixtral 8x7B, 8x22B | |
| Qwen 2 | Qwen 2, Qwen 2.5 | |
| Qwen 3 | Qwen 3, Qwen 3 MoE | |
| Qwen 2 VL | Qwen 2 VL, Qwen 2.5 VL, Qwen 2.5 Omni | ✓ |
| Gemma 2 | Gemma 2 | |
| Phi 3 | Phi 3, Phi 3 Vision | ✓ |
| DeepSeek V3 | DeepSeek V3 | |
| LLaVA | LLaVA 1.5, LLaVA-Next | ✓ |

## Installation

### Prerequisites

| Platform | Requirement |
|:---|:---|
| **NVIDIA GPU** | Compute Capability 7.5+ (Turing or later). CUDA 11.8+. |
| **AMD GPU** | ROCm version compatible with Triton. |
| **Intel CPU / GPU** | PyTorch and `intel_extension_for_pytorch` >= 2.4. |

### Install from Source (Recommended)

```bash
pip install git+https://github.com/BBC-Esq/AutoAWQ-Reborn.git
```

### Install Options

```bash
# Default (Triton-based inference)
pip install autoawq

# With precompiled CUDA kernels (must match your PyTorch version)
pip install autoawq[kernels]

# With Intel IPEX support for CPU inference
pip install autoawq[cpu]
```

## Quick Start

### Quantize a Model

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-14B-Instruct"
quant_path = "Qwen2.5-14B-Instruct-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### Run Inference

```python
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
from awq.utils.utils import get_best_device

device = get_best_device()
model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

model = AutoAWQForCausalLM.from_quantized(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the AWQ quantization algorithm in simple terms."},
]
inputs = tokenizer.apply_chat_template(
    prompt, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
).to(device)

model.generate(**inputs, do_sample=True, max_new_tokens=256, streamer=streamer)
```

## GEMM vs. GEMV

AutoAWQ-Reborn provides two primary quantized inference modes:

| Mode | Best For | Notes |
|:---|:---|:---|
| **GEMV** | Single-request inference (batch size 1) | ~20% faster than GEMM for single requests |
| **GEMM** | Batch inference and long contexts | Faster than FP16 at batch sizes below 8 |

At small batch sizes, inference is **memory-bound** — quantized weights are 3x smaller and move through memory faster. At larger batch sizes, inference becomes **compute-bound** and the dequantization overhead (INT4 → FP16) can reduce the speed advantage.

## Evaluation

AutoAWQ-Reborn includes a built-in evaluation suite:

```bash
# Perplexity (WikiText)
python examples/eval.py --model_path <model>

# MMLU
python examples/eval.py --model_path <model> --tasks mmlu --n_shot 5 --batch_size 4

# HumanEval
python examples/eval.py --model_path <model> --tasks humaneval

# Benchmarking
python examples/benchmark.py --model_path <model> --batch_size 1
```

## Documentation

Full documentation is available in the [`docs/`](docs/) directory and covers:

- [Installation and configuration](docs/index.md)
- [Quantization and inference examples](docs/examples.md)
- [API reference](docs/reference/index.md)

## Acknowledgments

AutoAWQ-Reborn is built on the work of many contributors:

- **[Casper Hansen](https://github.com/casper-hansen)** — creator of the original [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) project
- **[MIT HAN Lab](https://github.com/mit-han-lab/llm-awq)** — authors of the AWQ algorithm
- **[vLLM Project](https://github.com/vllm-project/vllm)** — for integrating AWQ support into production serving
- All open-source contributors who submitted patches, model support, and bug fixes to the original project

## Citation

If you use AWQ in your research, please cite the original paper:

```bibtex
@article{lin2023awq,
    title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
    author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
    journal={arXiv},
    year={2023}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
