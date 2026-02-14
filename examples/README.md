# AutoAWQ-Reborn Examples

This folder contains runnable example scripts for common workflows. For comprehensive documentation with additional examples and explanations, see the [full documentation](../docs/examples.md).

| Script | Description |
|:---|:---|
| `quantize.py` | Quantize a pretrained model to 4-bit AWQ format |
| `generate.py` | Load a quantized model and generate text with streaming |
| `eval.py` | Evaluate quantized models (perplexity, MMLU, HumanEval, etc.) |
| `benchmark.py` | Benchmark prefill and decode performance across context lengths |
| `train.py` | Fine-tune a quantized model with PEFT |
| `cli.py` | Command-line interface for common operations |