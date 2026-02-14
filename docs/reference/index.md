# API Reference

This page documents the primary public classes and methods in AutoAWQ-Reborn. The API reference is auto-generated from source code docstrings.

---

## Core Classes

### AutoAWQForCausalLM

The main entry point for loading and quantizing models. This class provides factory methods that automatically detect the model architecture and instantiate the appropriate model-specific class.

::: awq.models.auto.AutoAWQForCausalLM

---

### BaseAWQForCausalLM

The base class that all model-specific AWQ implementations inherit from. Contains the core quantization, saving, and loading logic.

::: awq.models.base.BaseAWQForCausalLM

---

## Quantization

### AwqQuantizer

The quantizer class that implements the AWQ algorithm. Can be extended to create custom quantizers for models that require special calibration handling (e.g., vision-language models).

::: awq.quantize.quantizer.AwqQuantizer

---

### AwqConfig

Configuration dataclass for quantization parameters.

::: awq.models._config.AwqConfig

---

## Evaluation

### Evaluation Functions

Built-in evaluation utilities for measuring quantized model quality.

::: awq.evaluation.eval_utils.evaluate_perplexity

::: awq.evaluation.eval_utils.eval_mmlu

::: awq.evaluation.eval_utils.eval_librispeech

::: awq.evaluation.humaneval_utils.eval_humaneval

::: awq.evaluation.kl_divergence.eval_kl_divergence

---

## Utilities

### Device Detection

::: awq.utils.utils.get_best_device

---

## Model Implementations

Each supported architecture has a dedicated AWQ implementation class. These are typically used through `AutoAWQForCausalLM` rather than directly:

| Class | Module | Architecture |
|:---|:---|:---|
| `LlamaAWQForCausalLM` | `awq.models.llama` | LLaMA family |
| `MistralAWQForCausalLM` | `awq.models.mistral` | Mistral |
| `MixtralAWQForCausalLM` | `awq.models.mixtral` | Mixtral |
| `Qwen2AWQForCausalLM` | `awq.models.qwen2` | Qwen 2 / 2.5 |
| `Qwen3AWQForCausalLM` | `awq.models.qwen3` | Qwen 3 |
| `Qwen3MoeAWQForCausalLM` | `awq.models.qwen3_moe` | Qwen 3 MoE |
| `Gemma2AWQForCausalLM` | `awq.models.gemma2` | Gemma 2 |
| `Phi3AWQForCausalLM` | `awq.models.phi3` | Phi 3 |
| `Phi3VAWQForCausalLM` | `awq.models.phi3_v` | Phi 3 Vision |
| `DeepseekV3AWQForCausalLM` | `awq.models.deepseek_v3` | DeepSeek V3 |
| `LlavaAWQForCausalLM` | `awq.models.llava` | LLaVA 1.5 |
| `LlavaNextAWQForCausalLM` | `awq.models.llava_next` | LLaVA-Next |
| `Qwen2VLAWQForCausalLM` | `awq.models.qwen2vl` | Qwen 2 VL |
| `Qwen2_5_VLAWQForCausalLM` | `awq.models.qwen2_5_vl` | Qwen 2.5 VL |
| `Qwen2_5_OmniAWQForConditionalGeneration` | `awq.models.qwen2_5_omni` | Qwen 2.5 Omni |

---

## Quantized Linear Layers

These classes implement the low-level quantized matrix operations for each backend:

| Class | Module | Backend |
|:---|:---|:---|
| `WQLinear_GEMM` | `awq.modules.linear.gemm` | GEMM (default) |
| `WQLinear_GEMV` | `awq.modules.linear.gemv` | GEMV |
| `WQLinear_GEMVFast` | `awq.modules.linear.gemv_fast` | GEMV Fast |
| `WQLinear_Exllama` | `awq.modules.linear.exllama` | ExLlama |
| `WQLinear_ExllamaV2` | `awq.modules.linear.exllamav2` | ExLlama V2 |
| `WQLinear_Marlin` | `awq.modules.linear.marlin` | Marlin |
| `WQLinear_IPEX` | `awq.modules.linear.gemm_ipex` | Intel IPEX |

---

## Fused Modules

Optimized fused operation classes for accelerated inference:

| Class | Module | Purpose |
|:---|:---|:---|
| `QuantAttentionFused` | `awq.modules.fused.attn` | Fused multi-head attention |
| `FasterTransformerRMSNorm` | `awq.modules.fused.norm` | Optimized RMS normalization |
| `FusedSparseMoeBlock` | `awq.modules.fused.moe` | Fused Mixture-of-Experts |
| `LlamaLikeBlock` | `awq.modules.fused.block` | Fused transformer block (LLaMA-like) |
| `MixtralBlock` | `awq.modules.fused.block` | Fused transformer block (Mixtral) |
| `Gemma2LikeBlock` | `awq.modules.fused.block` | Fused transformer block (Gemma 2) |
| `Phi3Block` | `awq.modules.fused.block` | Fused transformer block (Phi 3) |
| `WindowedCache` | `awq.modules.fused.cache` | Windowed KV cache implementation |
