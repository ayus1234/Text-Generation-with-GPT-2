# Text Generation with GPT-2

This project provides tools for fine-tuning and using GPT-2 for text generation with various decoding methods.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Demo

To quickly see a comparison of different text generation methods:

```bash
python text_generation_demo.py
```

This script will generate text using various decoding strategies:
1. Greedy Search
2. Beam Search
3. Beam Search with No-Repeat N-Gram
4. Sampling
5. Top-K Sampling
6. Top-p (Nucleus) Sampling
7. Multiple Sequences with Top-p

## Fine-tuning GPT-2

Fine-tune GPT-2 on your custom text dataset:

```bash
python finetune_gpt2.py --train_file sample_data.txt --epochs 3 --generate_samples
```

### Fine-tuning Options

```
--model_name MODEL_NAME   Model to fine-tune (gpt2, gpt2-medium, etc.)
--train_file TRAIN_FILE   Path to training file (required)
--block_size BLOCK_SIZE   Block size for dataset preparation (default: 128)
--output_dir OUTPUT_DIR   Directory to save the fine-tuned model (default: ./fine_tuned_model)
--batch_size BATCH_SIZE   Training batch size (default: 4)
--epochs EPOCHS           Number of training epochs (default: 3)
--learning_rate LEARNING_RATE
                        Learning rate (default: 5e-5)
--save_steps SAVE_STEPS   Save checkpoint every X steps (default: 1000)
--seed SEED               Random seed (default: 42)
--generate_samples        Generate sample text after training
--prompt PROMPT           Prompt for sample text generation (default: "Once upon a time")
--num_samples NUM_SAMPLES Number of samples to generate (default: 3)
```

## Generating Text from a Fine-tuned Model

After fine-tuning, use the specialized script to generate text from your fine-tuned model:

```bash
python generate_from_finetuned.py --model_path ./fine_tuned_model --prompt "Once upon a time"
```

### Generation Options

```
--model_path MODEL_PATH   Path to the fine-tuned model (required)
--prompt PROMPT           Prompt to start text generation (default: "Once upon a time")
--max_length MAX_LENGTH   Maximum length of generated text (default: 200)
--num_sequences NUM_SEQ   Number of sequences to generate (default: 3)
--decoding DECODING       Decoding strategy (default: top_p)
--temperature TEMP        Temperature for sampling (default: 0.7)
--top_k TOP_K             Top-k sampling parameter (default: 50)
--top_p TOP_P             Top-p sampling parameter (default: 0.9)
--no_repeat_ngram SIZE    Size of n-grams that cannot be repeated (default: 0)
--num_beams NUM_BEAMS     Number of beams for beam search (default: 5)
```

## Advanced Text Generation

For more control over text generation, use the main script:

```bash
python gpt2_text_generation.py --task generate --prompt "Once upon a time" --decoding top_p
```

See [gpt2_text_generation.py](gpt2_text_generation.py) for all available options.

## Decoding Methods

This project implements several decoding strategies:

1. **Greedy Search**: Always selects the most probable next token. Fast but often produces repetitive text.
   ```
   --decoding greedy
   ```

2. **Beam Search**: Keeps track of the most likely word sequences. Better than greedy but can still be repetitive.
   ```
   --decoding beam --num_beams 5
   ```

3. **Beam Search with n-gram penalty**: Prevents repetition of n-grams.
   ```
   --decoding beam --num_beams 5 --no_repeat_ngram 2
   ```

4. **Sampling**: Randomly selects the next token based on probability distribution. Introduces diversity.
   ```
   --decoding sample --temperature 0.7
   ```

5. **Top-K Sampling**: Samples from the K most likely next tokens. Good balance of coherence and diversity.
   ```
   --decoding top_k --top_k 50 --temperature 0.7
   ```

6. **Top-p (Nucleus) Sampling**: Samples from the smallest set of tokens whose cumulative probability exceeds threshold p. Often produces the most human-like text.
   ```
   --decoding top_p --top_p 0.9 --temperature 0.7
   ```

7. **Beam Search with Sampling**: Combines beam search with sampling for a mix of coherence and diversity.
   ```
   --decoding beam_sample --num_beams 5 --temperature 0.7 --top_p 0.9
   ```

## About Decoding Methods

- **Temperature**: Controls randomness. Lower values make the model more confident/deterministic; higher values make it more random.
- **Top-K**: Limits sampling to the K most likely next words.
- **Top-p**: Dynamically limits sampling to the smallest set of words whose cumulative probability exceeds p.
- **No-repeat n-gram**: Prevents the same n-gram from appearing twice in the generated text.

## Sample Data

The repository includes a small sample dataset (`sample_data.txt`) with fantasy-themed text for fine-tuning demonstrations.

## Project Structure

- `gpt2_text_generation.py` - Main script with comprehensive options for training and generation
- `text_generation_demo.py` - Quick demo of different text generation methods
- `finetune_gpt2.py` - Specialized script for fine-tuning GPT-2 on custom data
- `generate_from_finetuned.py` - Script for generating text from a fine-tuned model
- `sample_data.txt` - Sample dataset for fine-tuning experiments 