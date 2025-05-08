# Text Generation with GPT-2: Detailed Documentation

This document provides in-depth explanations and advanced usage examples for the Text Generation with GPT-2 project.

## Table of Contents
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [GPT-2 Model Variants](#gpt-2-model-variants)
- [Fine-tuning Process](#fine-tuning-process)
- [Decoding Methods in Detail](#decoding-methods-in-detail)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/ayus1234/Text_Generation_with_GPT_2.git
cd Text_Generation_with_GPT_2

# Install the package
pip install -e .
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/ayus1234/Text_Generation_with_GPT_2.git
cd Text_Generation_with_GPT_2

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

The project is structured around several key components:

1. **Model Loading**: Handles loading pre-trained GPT-2 models of various sizes.
2. **Dataset Preparation**: Processes text data for fine-tuning.
3. **Training Module**: Manages the fine-tuning process.
4. **Generation Module**: Implements different text generation strategies.
5. **CLI Interface**: Provides command-line tools for easy interaction.

Each script in the project corresponds to one or more of these components:

- `gpt2_text_generation.py`: Comprehensive script that incorporates all components.
- `text_generation_demo.py`: Simplified interface focusing on the generation module.
- `finetune_gpt2.py`: Specializes in the training module.
- `generate_from_finetuned.py`: Focuses on the generation module for fine-tuned models.

## GPT-2 Model Variants

GPT-2 comes in several sizes. This project supports all of them:

| Model Name    | Parameters | Layers | Hidden Size | Attention Heads |
|---------------|------------|--------|-------------|-----------------|
| gpt2          | 124M       | 12     | 768         | 12              |
| gpt2-medium   | 355M       | 24     | 1024        | 16              |
| gpt2-large    | 774M       | 36     | 1280        | 20              |
| gpt2-xl       | 1.5B       | 48     | 1600        | 25              |

To use a specific model variant, specify it with the `--model_name` parameter:

```bash
python finetune_gpt2.py --model_name gpt2-medium --train_file sample_data.txt
```

## Fine-tuning Process

### Dataset Preparation

The fine-tuning process starts with preparing your text dataset:

1. Create a text file with your training data
2. Each line or paragraph should be a complete example
3. The dataset should ideally contain 1,000+ examples for meaningful fine-tuning

### Hyperparameters

The key hyperparameters for fine-tuning are:

- **Learning Rate**: Controls how quickly the model adapts to your data. Lower values (1e-5 to 5e-5) are typically better.
- **Batch Size**: Number of examples processed simultaneously. Limited by GPU memory. Start with 4-8.
- **Epochs**: Number of passes through the dataset. For small datasets, 3-5 epochs is often sufficient.
- **Block Size**: Context window size. Depends on the content length. Default is 128 tokens.

### Example Fine-tuning Process

```bash
# Basic fine-tuning
python finetune_gpt2.py --train_file my_dataset.txt --epochs 3

# Advanced fine-tuning with more hyperparameters
python finetune_gpt2.py \
  --model_name gpt2-medium \
  --train_file my_dataset.txt \
  --output_dir ./my_custom_model \
  --block_size 256 \
  --batch_size 8 \
  --epochs 5 \
  --learning_rate 2e-5 \
  --save_steps 500 \
  --seed 42
```

### Fine-tuning Output

The fine-tuning process will create:

1. A directory with the fine-tuned model
2. Checkpoint directories for intermediate saves
3. Training logs showing loss over time

## Decoding Methods in Detail

### Greedy Search

- **Algorithm**: At each step, select the token with the highest probability.
- **Pros**: Fast, deterministic.
- **Cons**: Often produces repetitive text and can get stuck in loops.
- **Best for**: Short, factual completions where creativity isn't important.

```python
# Implementation detail
next_token_id = torch.argmax(next_token_logits, dim=-1)
```

### Beam Search

- **Algorithm**: Maintains top-K beam hypotheses at each step, expanding all possibilities.
- **Pros**: Often produces more coherent text than greedy search.
- **Cons**: Still prone to repetition, can be computationally expensive.
- **Best for**: Tasks where correctness is more important than diversity (e.g., summarization).

```python
# Key parameters
num_beams = 5         # Number of beams to track
early_stopping = True  # Stop when all beams reach EOS
```

### Sampling

- **Algorithm**: Randomly samples the next token from the probability distribution.
- **Pros**: Introduces diversity and creativity.
- **Cons**: Can produce incoherent or nonsensical text.
- **Best for**: Creative applications where diversity is valued.

```python
# Implementation with temperature
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

### Top-K Sampling

- **Algorithm**: Limits sampling to the K most likely tokens.
- **Pros**: Balance between coherence and diversity.
- **Cons**: K is a fixed hyperparameter regardless of the distribution shape.
- **Best for**: General text generation where quality and diversity are both important.

```python
# Implementation
topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
probs = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_probs)
probs = probs / probs.sum(dim=-1, keepdim=True)
next_token = torch.multinomial(probs, num_samples=1)
```

### Top-p (Nucleus) Sampling

- **Algorithm**: Samples from smallest set of tokens whose cumulative probability exceeds threshold p.
- **Pros**: Dynamically adjusts the sampling pool based on the distribution.
- **Cons**: Selecting the optimal p value can be challenging.
- **Best for**: High-quality creative text generation.

```python
# Implementation
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0
probs.scatter_(-1, sorted_indices, sorted_probs * (~sorted_indices_to_remove).float())
probs = probs / probs.sum(dim=-1, keepdim=True)
next_token = torch.multinomial(probs, num_samples=1)
```

## Advanced Usage

### Combining Methods

You can combine multiple decoding strategies for better results:

```bash
# Beam search with no-repeat n-gram penalty
python generate_from_finetuned.py \
  --model_path ./fine_tuned_model \
  --decoding beam \
  --num_beams 5 \
  --no_repeat_ngram 2

# Top-K and Top-p combined
python generate_from_finetuned.py \
  --model_path ./fine_tuned_model \
  --decoding top_p \
  --top_p 0.9 \
  --top_k 50
```

### Controlling Generation Length

Control the length of generated text:

```bash
python generate_from_finetuned.py \
  --model_path ./fine_tuned_model \
  --prompt "Once upon a time" \
  --max_length 300
```

### Generating Multiple Outputs

Generate multiple different outputs from the same prompt:

```bash
python generate_from_finetuned.py \
  --model_path ./fine_tuned_model \
  --prompt "Once upon a time" \
  --num_sequences 5
```

## Best Practices

### For Fine-tuning

1. **Start Small**: Begin with the smallest GPT-2 model (124M) for quick iterations.
2. **Quality Data**: Use high-quality text data relevant to your target domain.
3. **Validate**: Generate samples during training to validate progress.
4. **Save Checkpoints**: Save checkpoints regularly to resume training if needed.
5. **Hyperparameter Tuning**: Experiment with learning rates between 1e-5 and 5e-5.

### For Text Generation

1. **Right Decoding**: Choose the appropriate decoding method for your use case:
   - Factual/precise text: Beam search
   - Creative text: Top-p sampling
   - Balanced approach: Top-K + Top-p
2. **Temperature**: Adjust temperature based on desired randomness:
   - Lower (0.3-0.5): More focused, deterministic
   - Medium (0.6-0.8): Balanced
   - Higher (0.9-1.2): More random, creative
3. **Repetition Control**: Use no-repeat ngram penalties to avoid loops.
4. **Prompt Engineering**: Craft your prompts carefully to guide the model.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Use a smaller model variant
   - Reduce batch size
   - Reduce block size
   - Use gradient accumulation

2. **Low Quality Output**
   - Increase dataset size
   - Train for more epochs
   - Try different decoding parameters
   - Check data quality

3. **Slow Training**
   - Use GPU acceleration
   - Reduce batch size
   - Decrease model size
   - Use gradient checkpointing

## Performance Tips

### Training Performance

- Use fp16 training on supported GPUs
- Enable gradient checkpointing for large models
- Use smaller block sizes if possible
- Train on a GPU with at least 8GB VRAM for optimal performance

### Generation Performance

- Batch your generation requests
- Use lower values for max_length when possible
- For interactive applications, set appropriate timeouts
- Consider beam size vs. generation quality tradeoffs 