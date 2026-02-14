# Examples

This page provides complete, working examples for quantization, inference, evaluation, and advanced workflows. Each example uses modern model references and current best practices.

---

## Quantization

### Basic Quantization

AWQ performs activation-aware weight quantization to 4-bit integers. The process requires calibration data (WikiText-2 is used by default) and typically takes 10-15 minutes for 7B models and up to an hour for 70B models.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-14B-Instruct"
quant_path = "Qwen2.5-14B-Instruct-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

**Quantization config parameters:**

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `zero_point` | `bool` | `True` | Enable zero-point quantization |
| `q_group_size` | `int` | `128` | Quantization group size (some models require 64) |
| `w_bit` | `int` | `4` | Weight bit width |
| `version` | `str` | `"GEMM"` | Backend version: `"GEMM"`, `"GEMV"`, or `"Marlin"` |

!!! note
    Some models (e.g., Falcon) are only compatible with `q_group_size=64`. Marlin requires `zero_point=False`.

### Custom Calibration Data

You can provide your own calibration data for domain-specific quantization. The calibration data should be a list of text strings.

```python
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-7B-Instruct"
quant_path = "Qwen2.5-7B-Instruct-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Custom calibration data loaders
def load_dolly():
    data = load_dataset("databricks/databricks-dolly-15k", split="train")

    def concatenate_data(x):
        return {"text": x["instruction"] + "\n" + x["context"] + "\n" + x["response"]}

    concatenated = data.map(concatenate_data)
    return [text for text in concatenated["text"]]

def load_wikitext():
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [text for text in data["text"] if text.strip() != "" and len(text.split(" ")) > 20]

# Quantize with custom data
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### Long-Context Quantization

For long-context models, use a high-quality dataset filtered by token length and tune memory parameters to avoid OOM errors.

```python
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-7B-Instruct"
quant_path = "Qwen2.5-7B-Instruct-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_cosmopedia():
    data = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
    data = data.filter(lambda x: x["text_token_length"] >= 2048)
    return [text for text in data["text"]]

# Quantize with long-context parameters
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_cosmopedia(),
    n_parallel_calib_samples=32,
    max_calib_samples=128,
    max_calib_seq_len=4096,
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

**Memory management parameters:**

| Parameter | Type | Description |
|:---|:---|:---|
| `n_parallel_calib_samples` | `int` | Number of samples processed in parallel. Lower values reduce GPU VRAM usage but offload to system RAM. |
| `max_calib_samples` | `int` | Total calibration samples. 128-256 is typically sufficient. |
| `max_calib_seq_len` | `int` | Maximum sequence length for calibration samples. |

### Quantization for Code Models

For code-oriented models, use a coding-specific calibration dataset:

```python
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
quant_path = "deepseek-coder-v2-lite-instruct-awq"
quant_config = {"zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_openhermes_coding():
    data = load_dataset("alvarobartt/openhermes-preferences-coding", split="train")
    samples = []
    for sample in data:
        responses = [
            f'{response["role"]}: {response["content"]}'
            for response in sample["chosen"]
        ]
        samples.append("\n".join(responses))
    return samples

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_openhermes_coding(),
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### Vision-Language Model Quantization

AutoAWQ-Reborn supports quantizing vision-language models such as LLaVA:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "llava-hf/llama3-llava-next-8b-hf"
quant_path = "llama3-llava-next-8b-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### GGUF Export

You can apply AWQ scales without packing weights, then convert to GGUF format for use with llama.cpp:

```python
import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "mistralai/Mistral-7B-v0.1"
quant_path = "mistral-awq"
llama_cpp_path = "/workspace/llama.cpp"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 6, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Apply AWQ scales without packing (FP16 output with AWQ scales applied)
model.quantize(
    tokenizer,
    quant_config=quant_config,
    export_compatible=True,
)

# Save as FP16 with AWQ scales
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Convert to GGUF
llama_cpp_method = "q4_K_M"
convert_cmd_path = os.path.join(llama_cpp_path, "convert.py")
quantize_cmd_path = os.path.join(llama_cpp_path, "quantize")

subprocess.run(
    [f"python {convert_cmd_path} {quant_path} --outfile {quant_path}/model.gguf"],
    shell=True, check=True,
)
subprocess.run(
    [f"{quantize_cmd_path} {quant_path}/model.gguf {quant_path}/model_{llama_cpp_method}.gguf {llama_cpp_method}"],
    shell=True, check=True,
)
```

!!! warning
    Models exported with `export_compatible=True` cannot be loaded back into AutoAWQ for inference. The saved model is FP16 with AWQ scales applied and is intended for conversion to other formats.

---

## Inference

### GPU Inference

Load a quantized model and generate text with streaming output:

```python
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
from awq.utils.utils import get_best_device

device = get_best_device()
model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# Load model
model = AutoAWQForCausalLM.from_quantized(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Build prompt using chat template
prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the AWQ quantization algorithm in simple terms."},
]
inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(device)

# Generate
model.generate(**inputs, do_sample=True, max_new_tokens=256, streamer=streamer)
```

### GPU Inference with Fused Modules

Fused modules combine multiple operations into optimized kernels for faster inference:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# Load with fused layers enabled
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt_template = "[INST] {prompt} [/INST]"
prompt = "What are the key differences between GEMM and GEMV quantization?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt),
    return_tensors="pt",
).input_ids.cuda()

generation_output = model.generate(tokens, streamer=streamer, max_new_tokens=512)
```

### CPU Inference (Intel IPEX)

For CPU inference using Intel Extension for PyTorch:

```python
from awq import AutoAWQForCausalLM

quant_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# Load with IPEX backend
model = AutoAWQForCausalLM.from_quantized(quant_path, use_ipex=True)
```

### Hugging Face Transformers Integration

AWQ models can also be loaded directly through the Transformers `AutoModelForCausalLM` class (requires AutoAWQ to be installed):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "casperhansen/mistral-7b-instruct-v0.1-awq"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text = "[INST] What are the basic steps to use the Huggingface transformers library? [/INST]"
tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()

generation_output = model.generate(tokens, streamer=streamer, max_new_tokens=512)
```

!!! info
    When loading through Transformers, not all models will have access to fused modules. For full performance, load through `AutoAWQForCausalLM` directly. See the [Transformers AWQ documentation](https://huggingface.co/docs/transformers/main/en/quantization/awq) for more details.

### vLLM Integration

AWQ models are fully supported by [vLLM](https://github.com/vllm-project/vllm) for high-throughput serving:

```python
import asyncio
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

model_path = "casperhansen/mixtral-instruct-awq"

prompt = (
    "You're standing on the surface of the Earth. "
    "You walk one mile south, one mile west and one mile north. "
    "You end up exactly where you started. Where are you?"
)
prompt_template = "[INST] {prompt} [/INST]"

sampling_params = SamplingParams(
    repetition_penalty=1.1,
    temperature=0.8,
    max_tokens=512,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

engine_args = AsyncEngineArgs(
    model=model_path,
    quantization="awq",
    dtype="float16",
    max_model_len=512,
    enforce_eager=True,
    disable_log_requests=True,
    disable_log_stats=True,
)


async def generate(model: AsyncLLMEngine, tokenizer: PreTrainedTokenizer):
    tokens = tokenizer(prompt_template.format(prompt=prompt)).input_ids

    outputs = model.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=1,
        prompt_token_ids=tokens,
    )

    print("\n** Starting generation!\n")
    last_index = 0
    async for output in outputs:
        print(output.outputs[0].text[last_index:], end="", flush=True)
        last_index = len(output.outputs[0].text)
    print("\n\n** Finished generation!\n")


if __name__ == "__main__":
    model = AsyncLLMEngine.from_engine_args(engine_args)
    asyncio.run(generate(model, tokenizer))
```

---

## Vision-Language Model Inference

### LLaVA

Load a quantized LLaVA model for image understanding:

```python
import torch
import requests
from PIL import Image
from awq import AutoAWQForCausalLM
from transformers import AutoProcessor, TextStreamer

quant_path = "casperhansen/llama3-llava-next-8b-awq"
model = AutoAWQForCausalLM.from_quantized(quant_path)
processor = AutoProcessor.from_pretrained(quant_path)
streamer = TextStreamer(processor, skip_prompt=True)

prompt = """\
<|im_start|>system\nAnswer the questions.<|im_end|>
<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>
<|im_start|>assistant
"""

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(prompt, image, return_tensors="pt").to(0, torch.float16)

generation_output = model.generate(**inputs, max_new_tokens=512, streamer=streamer)
```

### Qwen2 VL

Run inference with the Qwen2 VL vision-language model:

```python
from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, TextStreamer

quant_path = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(quant_path)
processor = AutoProcessor.from_pretrained(quant_path)
streamer = TextStreamer(processor, skip_prompt=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generation_output = model.generate(**inputs, max_new_tokens=512, streamer=streamer)
```

---

## Advanced Quantization

### Custom Quantizer (Qwen2 VL)

For vision-language models that require special input processing during calibration, you can extend the `AwqQuantizer` class:

```python
import torch
import torch.nn as nn
from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

model_path = "Qwen/Qwen2-VL-7B-Instruct"
quant_path = "qwen2-vl-7b-instruct"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(
    model_path, attn_implementation="flash_attention_2"
)


class Qwen2VLAwqQuantizer(AwqQuantizer):
    """Custom quantizer that handles multimodal inputs during calibration."""

    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.awq_model.get_model_layers(self.model)
        samples = self.calib_data

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)
                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit

        def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
            def get_device(obj: torch.Tensor | nn.Module):
                if isinstance(obj, torch.Tensor):
                    return obj.device
                return next(obj.parameters()).device

            if get_device(obj) != device:
                obj = obj.to(device)
            return obj

        modules[0] = Catcher(modules[0])
        for k, v in samples.items():
            if isinstance(v, (torch.Tensor, nn.Module)):
                samples[k] = move_to_device(v, best_device)
        try:
            self.model(**samples)
        except ValueError:
            pass
        finally:
            for k, v in samples.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    samples[k] = move_to_device(v, "cpu")
        modules[0] = modules[0].module

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps


# Prepare multimodal calibration data
def prepare_dataset(n_sample: int = 8) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]"
    )
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "generate a caption for this image"},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]


dataset = prepare_dataset()

text = model.processor.apply_chat_template(
    dataset, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(dataset)
inputs = model.processor(
    text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
)

model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=Qwen2VLAwqQuantizer)

model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
```

### Custom Quantizer (MiniCPM3)

Some models require modified weight clipping during quantization. This example shows how to customize the clipping behavior:

```python
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import AwqQuantizer, clear_memory


class CPM3AwqQuantizer(AwqQuantizer):
    """Custom quantizer with modified weight clipping for MiniCPM3."""

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64
        if org_w_shape[0] % oc_batch_size != 0:
            oc_batch_size = org_w_shape[0]
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w, cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)


model_path = "openbmb/MiniCPM3-4B"
quant_path = "minicpm3-4b-awq"
quant_config = {"zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config, quantizer_cls=CPM3AwqQuantizer)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

---

## Evaluation

AutoAWQ-Reborn provides built-in evaluation tools accessible through the `examples/eval.py` script or programmatically.

### Command-Line Evaluation

```bash
# Perplexity on WikiText-2 (quantized model)
python examples/eval.py --model_path casperhansen/mistral-7b-instruct-v0.1-awq

# Perplexity on WikiText-2 (FP16 baseline)
python examples/eval.py --use_pretrained --model_path mistralai/Mistral-7B-Instruct-v0.2

# MMLU benchmark
python examples/eval.py --model_path TheBloke/zephyr-7B-beta-AWQ --tasks mmlu --n_shot 1 --batch_size 4

# HumanEval code generation
python examples/eval.py --model_path <model> --tasks humaneval

# KL divergence between quantized and original
python examples/eval.py --model_path <model> --tasks kldiv

# LibriSpeech (speech recognition models)
python examples/eval.py --model_path <model> --tasks librispeech
```

### Available Evaluation Tasks

| Task | Description | Metric |
|:---|:---|:---|
| `wikitext` | Language modeling on WikiText-2 | Perplexity |
| `mmlu` | Massive Multitask Language Understanding | Accuracy |
| `humaneval` | Code generation benchmark | Pass@k |
| `kldiv` | KL divergence vs. reference model | KL divergence |
| `librispeech` | Speech recognition evaluation | Word Error Rate (WER) |

You can also use any task supported by the [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) by passing task names directly.

---

## Benchmarking

Use the benchmarking script to measure prefill and decode performance:

```bash
# Basic benchmark
python examples/benchmark.py --model_path <model> --batch_size 1

# Benchmark with Hugging Face generate method
python examples/benchmark.py --model_path <model> --batch_size 1 --generator hf

# Benchmark FP16 baseline
python examples/benchmark.py --model_path <model> --pretrained --batch_size 1
```

The benchmark script tests multiple context lengths (32 to 4096 tokens) and reports:

- **Prefill tokens/s** — speed of processing the input context
- **Decode tokens/s** — speed of generating new tokens
- **Memory (VRAM)** — peak GPU memory usage
