# AutoAWQ-Reborn

AutoAWQ-Reborn is the actively maintained successor to [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), providing fast and accessible 4-bit quantization for large language models using the AWQ (Activation-aware Weight Quantization) algorithm. This documentation covers installation, configuration, quantization workflows, inference, and the full API reference.

!!! info "Continuing the Legacy"
    AutoAWQ-Reborn builds on the original AutoAWQ by Casper Hansen. The original project was deprecated, and this fork carries development forward with support for the latest models, Transformers versions, and Python releases.

## Performance Overview

AWQ quantization delivers substantial improvements over FP16 inference:

- **3x memory reduction** — 4-bit weights are roughly one-quarter the size of FP16
- **Up to 3x faster inference** — smaller weights move through GPU memory bandwidth faster
- **Minimal quality loss** — AWQ preserves the most salient weights based on activation patterns

## Installation

### System Requirements

=== "NVIDIA GPU"

    - **GPU**: Compute Capability 7.5 or later (Turing, Ampere, Ada Lovelace, Hopper)
    - **CUDA**: 11.8 or later
    - **Driver**: Compatible with your CUDA version

=== "AMD GPU"

    - **ROCm**: Version compatible with Triton
    - Inference runs through ExLlama V2 kernels (fused layers not supported)

=== "Intel CPU / GPU"

    - **PyTorch**: 2.4 or later
    - **IPEX**: `intel_extension_for_pytorch` >= 2.4
    - Alternatively, use [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton) for GPU

### Install from Source

For the latest features and model support, install directly from the repository:

```bash
pip install git+https://github.com/BBC-Esq/AutoAWQ-Reborn.git
```

### Install from PyPI

```bash
# Default — uses Triton for inference kernels
pip install autoawq

# With precompiled CUDA kernels (must match your PyTorch version)
pip install autoawq[kernels]

# With Intel IPEX support for CPU inference
pip install autoawq[cpu]

# With evaluation dependencies
pip install autoawq[eval]
```

### Dependencies

AutoAWQ-Reborn requires the following core packages (installed automatically):

| Package | Minimum Version | Purpose |
|:---|:---|:---|
| `torch` | — | PyTorch backend |
| `triton` | — | Default inference kernels |
| `transformers` | >= 4.45.0 | Model loading and tokenization |
| `tokenizers` | >= 0.12.1 | Fast tokenization |
| `accelerate` | — | Multi-GPU and device mapping |
| `datasets` | >= 2.20 | Calibration data loading |
| `huggingface_hub` | >= 1.0.0 | Model downloads |
| `typing_extensions` | >= 4.8.0 | Type hint support |
| `zstandard` | — | Compressed model support |

## Supported Models

AutoAWQ-Reborn supports the following architectures for both quantization and inference:

### Text Models

| Model | Architecture | Fused Modules | MoE |
|:---|:---|:---:|:---:|
| LLaMA (2, 3, 3.1, Code Llama) | `LlamaForCausalLM` | ✓ | |
| Mistral | `MistralForCausalLM` | ✓ | |
| Mixtral | `MixtralForCausalLM` | ✓ | ✓ |
| Qwen 2 / 2.5 | `Qwen2ForCausalLM` | ✓ | |
| Qwen 3 | `Qwen3ForCausalLM` | ✓ | |
| Qwen 3 MoE | `Qwen3MoeForCausalLM` | | ✓ |
| Gemma 2 | `Gemma2ForCausalLM` | ✓ | |
| Phi 3 | `Phi3ForCausalLM` | ✓ | |
| DeepSeek V3 | `DeepseekV3ForCausalLM` | | ✓ |

### Vision-Language Models

| Model | Architecture | Notes |
|:---|:---|:---|
| LLaVA 1.5 | `LlavaForConditionalGeneration` | Image understanding |
| LLaVA-Next | `LlavaNextForConditionalGeneration` | Improved image reasoning |
| Phi 3 Vision | `Phi3VForCausalLM` | Multimodal Phi |
| Qwen 2 VL | `Qwen2VLForConditionalGeneration` | Video and image understanding |
| Qwen 2.5 VL | `Qwen2_5_VLForConditionalGeneration` | Enhanced vision-language |
| Qwen 2.5 Omni | `Qwen2_5_OmniForConditionalGeneration` | Audio, image, and video |

## Quantization Backends

AutoAWQ-Reborn supports multiple inference backends. The backend is selected at quantization time via the `version` parameter or at load time:

| Backend | Key | Best For | Platform |
|:---|:---|:---|:---|
| GEMM | `GEMM` | Batch inference, long contexts | CUDA |
| GEMV | `GEMV` | Single-request inference (batch size 1) | CUDA |
| GEMV Fast | — | Optimized single-request inference | CUDA |
| ExLlama | — | Legacy kernel support | CUDA |
| ExLlama V2 | — | High-performance inference, AMD ROCm | CUDA, ROCm |
| Marlin | `Marlin` | Optimized 4-bit inference | CUDA |
| IPEX | — | CPU inference | Intel CPU/GPU |

### Choosing a Backend

- **GEMM** is the default and recommended backend for most use cases. It handles both single requests and batched workloads efficiently.
- **GEMV** provides a ~20% speed advantage over GEMM when processing single requests (batch size 1), but does not scale well to larger batch sizes.
- **Marlin** offers highly optimized 4-bit inference and requires `zero_point=False` during quantization.
- **ExLlama V2** provides broad compatibility including AMD ROCm support.
- **IPEX** enables inference on Intel CPUs and GPUs.

## Fused Modules

Fused modules combine multiple operations (attention, normalization, feed-forward) into optimized single kernels for faster inference. They are enabled at load time:

```python
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
```

!!! warning "Fused Module Limitations"
    - The FasterTransformer-based acceleration is only compatible with **Linux**.
    - A custom KV cache is preallocated based on `max_seq_len` and `batch_size` — these cannot be changed after model creation.
    - The `past_key_values` returned by `model.generate()` are placeholder values and cannot be reused.

Fused module support varies by model:

| Fused Block Type | Models |
|:---|:---|
| `LlamaLikeBlock` | LLaMA, Mistral, Qwen 2, Qwen 3 |
| `MixtralBlock` | Mixtral |
| `Gemma2LikeBlock` | Gemma 2 |
| `Phi3Block` | Phi 3 |
| `FusedSparseMoeBlock` | Mixtral (MoE layers) |

## Platform-Specific Configuration

### AMD GPU (ROCm)

For AMD GPUs, use ExLlama V2 kernels without fused layers:

```python
model = AutoAWQForCausalLM.from_quantized(
    quant_path,
    fuse_layers=False,
    use_exllama_v2=True,
)
```

### Intel CPU (IPEX)

For CPU inference with Intel Extension for PyTorch:

```python
model = AutoAWQForCausalLM.from_quantized(
    quant_path,
    use_ipex=True,
)
```

!!! note
    Ensure your `intel_extension_for_pytorch` version matches your PyTorch version. IPEX 2.4 requires PyTorch 2.4.

## What's Next

- **[Examples](examples.md)** — Complete quantization, inference, and evaluation workflows
- **[API Reference](reference/index.md)** — Detailed documentation of all public classes and methods
