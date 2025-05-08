import os
import torch
import argparse
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TextDataset, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)

def load_model(model_name_or_path="gpt2", device="cuda"):
    """
    Load the GPT-2 model and tokenizer
    """
    print(f"Loading model: {model_name_or_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    
    # Handle special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    
    return model, tokenizer

def prepare_dataset(file_path, tokenizer, block_size=128):
    """
    Prepare a dataset for training from a text file
    """
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )
    
    return dataset, data_collator

def train_model(model, tokenizer, train_file, output_dir, 
                per_device_train_batch_size=4, 
                num_train_epochs=3, 
                save_steps=1000):
    """
    Fine-tune the GPT-2 model on a custom dataset
    """
    # Prepare dataset
    train_dataset, data_collator = prepare_dataset(train_file, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Start training
    print(f"Starting training on {train_file}")
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def generate_text(model, tokenizer, prompt, 
                  max_length=100, 
                  num_return_sequences=1,
                  decoding_strategy="greedy",
                  temperature=0.7,
                  top_k=0,
                  top_p=0.9,
                  no_repeat_ngram_size=0,
                  num_beams=1):
    """
    Generate text using various decoding strategies
    
    Parameters:
    - decoding_strategy: One of ["greedy", "beam", "sample", "top_k", "top_p", "beam_sample"]
    """
    # Encode prompt
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(model.device)
    
    # Set up generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "num_return_sequences": num_return_sequences,
    }
    
    # Configure decoding strategy
    if decoding_strategy == "greedy":
        # Default: greedy decoding
        pass
    elif decoding_strategy == "beam":
        gen_kwargs["num_beams"] = num_beams
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        gen_kwargs["early_stopping"] = True
    elif decoding_strategy == "sample":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    elif decoding_strategy == "top_k":
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_k"] = top_k
        gen_kwargs["temperature"] = temperature
    elif decoding_strategy == "top_p":
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = top_p
        gen_kwargs["temperature"] = temperature
    elif decoding_strategy == "beam_sample":
        gen_kwargs["do_sample"] = True
        gen_kwargs["num_beams"] = num_beams
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
    
    # Generate text
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        **gen_kwargs
    )
    
    # Decode generated sequences
    generated_texts = []
    for i, sequence in enumerate(output_sequences):
        # Remove the prompt from the generated sequence
        sequence = sequence[len(encoded_prompt[0]):]
        
        # Decode text
        text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
        generated_texts.append(text)
    
    return generated_texts

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Text Generation")
    
    # Task: train or generate
    parser.add_argument("--task", type=str, choices=["train", "generate"], required=True,
                        help="Task to perform: train or generate")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Model to load (gpt2, gpt2-medium, etc., or path to fine-tuned model)")
    
    # Training parameters
    parser.add_argument("--train_file", type=str, help="Path to training file")
    parser.add_argument("--output_dir", type=str, default="./trained_model",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt for text generation")
    parser.add_argument("--decoding", type=str, 
                        choices=["greedy", "beam", "sample", "top_k", "top_p", "beam_sample"],
                        default="greedy", 
                        help="Decoding strategy")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--num_sequences", type=int, default=1,
                        help="Number of sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--no_repeat_ngram", type=int, default=0,
                        help="Size of n-grams that cannot be repeated")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name)
    
    if args.task == "train":
        # Check if training file exists
        if not args.train_file or not os.path.exists(args.train_file):
            raise ValueError(f"Training file not found: {args.train_file}")
        
        # Train model
        train_model(
            model=model,
            tokenizer=tokenizer,
            train_file=args.train_file,
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs
        )
    
    elif args.task == "generate":
        # Check if prompt is provided
        if not args.prompt:
            args.prompt = "Once upon a time"
            print(f"No prompt provided, using default: '{args.prompt}'")
        
        # Generate text
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            num_return_sequences=args.num_sequences,
            decoding_strategy=args.decoding,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram,
            num_beams=args.num_beams
        )
        
        # Print generated texts
        print("\n" + "="*50)
        print(f"Generated {len(generated_texts)} text(s) using {args.decoding} decoding:")
        for i, text in enumerate(generated_texts):
            print(f"\n--- Sequence {i+1} ---")
            print(f"{args.prompt}{text}")
        print("="*50)

if __name__ == "__main__":
    main() 