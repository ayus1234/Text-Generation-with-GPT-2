import os
import argparse
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TextDataset, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)

def load_model(model_name="gpt2", device="cuda"):
    """
    Load the GPT-2 model and tokenizer
    """
    print(f"Loading model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
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
    print(f"Preparing dataset from {file_path} with block size {block_size}")
    
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
                save_steps=1000,
                block_size=128,
                learning_rate=5e-5,
                seed=42):
    """
    Fine-tune the GPT-2 model on a custom dataset
    """
    # Prepare dataset
    train_dataset, data_collator = prepare_dataset(train_file, tokenizer, block_size)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        save_total_limit=2,
        seed=seed,
        logging_steps=100,
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

def generate_sample_text(model, tokenizer, prompt="Once upon a time", num_samples=3):
    """
    Generate sample text to showcase the fine-tuned model
    """
    print("\nGenerating sample text with fine-tuned model:")
    print("="*50)
    
    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(model.device)
    
    # Generate text
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=100,
        num_return_sequences=num_samples,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
    )
    
    # Print generated text
    for i, sequence in enumerate(output_sequences):
        text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
        print(f"\nSample {i+1}:")
        print(text)
        print("-"*50)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on a custom dataset")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Model to fine-tune (gpt2, gpt2-medium, etc.)")
    
    # Dataset parameters
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training file")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for dataset preparation")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Generation parameters
    parser.add_argument("--generate_samples", action="store_true",
                        help="Generate sample text after training")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for sample text generation")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Check if training file exists
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name)
    
    # Train model
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_file=args.train_file,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    # Generate sample text
    if args.generate_samples:
        generate_sample_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main() 