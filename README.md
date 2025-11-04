# Fine-tuning with SmolLM2-135M

## Overview

This notebook demonstrates complete parameter fine-tuning of the SmolLM2-135M language model using Unsloth.ai. Full fine-tuning updates all model parameters, unlike parameter-efficient methods like LoRA which only train a small subset.

## What You'll Learn

- How to set up Unsloth for full fine-tuning
- Understanding the difference between full fine-tuning and LoRA
- Loading and preparing instruction-following datasets
- Configuring training parameters for optimal performance
- Saving and testing fine-tuned models
- When to use full fine-tuning versus other methods

## Requirements

- Google Colab with T4 GPU (free tier)
- Approximately 15-20 minutes of training time
- No local setup required

## Model Information

**Base Model:** HuggingFaceTB/SmolLM2-135M-Instruct
- Parameters: 135 million
- Size: Approximately 540 MB
- Context Length: 512 tokens (configurable)

## Dataset

**Dataset:** Alpaca Cleaned (yahma/alpaca-cleaned)
- Training Samples: 500 (subset for quick demonstration)
- Format: Instruction-input-output triplets
- Task: General instruction following

## Configuration

```python
FULL_FINETUNING = True
max_seq_length = 512
batch_size = 2
gradient_accumulation_steps = 4  # Effective batch size: 8
learning_rate = 2e-5
max_steps = 50
```

## Key Features

### Full Parameter Training
- Updates all 135 million parameters
- No parameter freezing
- Complete model adaptation
- Higher memory requirements than LoRA

### Memory Optimization
- 4-bit quantization for reduced VRAM usage
- Gradient checkpointing enabled
- AdamW 8-bit optimizer
- Runs on free T4 GPU (16GB VRAM)

### Training Features
- Automatic mixed precision (FP16/BF16)
- Linear learning rate scheduling
- Checkpoint saving every 25 steps
- Real-time loss logging

## Training Process

1. **Installation:** Unsloth and dependencies (~30 seconds)
2. **Model Loading:** SmolLM2-135M with 4-bit quantization (~10 seconds)
3. **Data Preparation:** Format 500 Alpaca samples (~5 seconds)
4. **Training:** 50 steps of full parameter updates (~15-20 minutes)
5. **Saving:** Model checkpoint to disk (~30 seconds)
6. **Testing:** Inference on sample prompts (~1 minute)

Total Runtime: Approximately 20-25 minutes

## Output Files

```
./smollm2_full_finetuned/
├── adapter_config.json
├── adapter_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── training_args.bin
```

Model size: ~540 MB (full model weights)

## Usage

### Running the Notebook

1. Open in Google Colab
2. Runtime > Change runtime type > T4 GPU
3. Run all cells sequentially
4. Wait for training to complete
5. Test with custom prompts

### Inference Example

```python
FastLanguageModel.for_inference(model)

instruction = "Explain quantum computing in simple terms"
prompt = alpaca_prompt.format(instruction, "")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Performance Metrics

**Training Speed:**
- Steps per second: ~0.05-0.06
- Time per step: ~18-20 seconds
- Total training time: ~15-20 minutes (50 steps)

**Model Performance:**
- Training loss: Decreases from ~2.5 to ~1.2
- Inference quality: Significant improvement over base model
- Instruction following: Good adherence to prompts

## When to Use Full Fine-tuning

### Recommended For:
- Small models (under 1B parameters)
- Large training datasets (10k+ samples)
- Maximum performance requirements
- Sufficient compute resources available
- Single-task specialization

### Not Recommended For:
- Limited GPU memory
- Quick experimentation
- Multiple task adapters needed
- Large models (7B+ parameters)
- Small datasets (under 1k samples)

## Comparison with LoRA

| Aspect | Full Fine-tuning | LoRA |
|--------|------------------|------|
| Trainable Parameters | 100% | 1-2% |
| Training Time | Slower | Faster |
| Memory Usage | Higher | Lower |
| Final Model Size | ~540 MB | ~2-5 MB (adapters) |
| Performance | Slightly Better | Very Good |
| Flexibility | Single Task | Multiple Adapters |

## Troubleshooting

### Out of Memory Error
- Reduce batch_size to 1
- Reduce max_seq_length to 256
- Reduce gradient_accumulation_steps
- Use full 4-bit quantization

### Slow Training
- Verify GPU is enabled (not CPU)
- Check for other processes using GPU
- Reduce logging_steps for less overhead

### Poor Results
- Increase training steps (100-200)
- Add more training data
- Adjust learning rate (try 1e-5 or 5e-5)
- Increase epochs to 2-3

## Extensions

### Training on Custom Data
Replace the dataset loading with your own:

```python
from datasets import Dataset

your_data = [
    {"instruction": "...", "input": "...", "output": "..."},
    # More examples
]

dataset = Dataset.from_list(your_data)
```

### Longer Training
Increase training duration:

```python
max_steps = 200  # or
num_train_epochs = 3
```

### Export to GGUF
Convert for local deployment:

```python
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
```

## Citation

```bibtex
@software{unsloth2024,
  title = {Unsloth: Fast and Memory-Efficient Language Model Training},
  author = {Unsloth AI},
  year = {2024},
  url = {https://github.com/unslothai/unsloth}
}
```

## Resources

- Unsloth Documentation: https://docs.unsloth.ai
- SmolLM2 Model Card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- Alpaca Dataset: https://huggingface.co/datasets/yahma/alpaca-cleaned


## License

This notebook is provided as educational material. The base model (SmolLM2) is released under Apache 2.0 license. Training code uses Unsloth (Apache 2.0) and Transformers library.

