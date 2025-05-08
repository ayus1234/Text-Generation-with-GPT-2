import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path):
    """
    Load a fine-tuned GPT-2 model and tokenizer
    """
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, 
                  max_length=100, 
                  num_return_sequences=1,
                  decoding_strategy="top_p",
                  temperature=0.7,
                  top_k=0,
                  top_p=0.9,
                  no_repeat_ngram_size=0,
                  num_beams=1):
    """
    Generate text using various decoding strategies
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
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
        input_ids=input_ids,
        **gen_kwargs
    )
    
    # Process generated sequences
    generated_texts = []
    for i, output in enumerate(output_sequences):
        text = tokenizer.decode(output, clean_up_tokenization_spaces=True)
        generated_texts.append(text)
    
    return generated_texts

def main():
    parser = argparse.ArgumentParser(description="Generate text from a fine-tuned GPT-2 model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt to start text generation")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum length of generated text")
    parser.add_argument("--num_sequences", type=int, default=3,
                        help="Number of sequences to generate")
    parser.add_argument("--decoding", type=str, default="top_p",
                        choices=["greedy", "beam", "sample", "top_k", "top_p", "beam_sample"],
                        help="Decoding strategy to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (higher = more random)")
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
    model, tokenizer = load_model(args.model_path)
    
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
    
    # Print generated text
    print("\n" + "="*50)
    print(f"Generated {len(generated_texts)} text(s) using {args.decoding} decoding:")
    for i, text in enumerate(generated_texts):
        print(f"\n--- Sequence {i+1} ---")
        print(text)
    print("="*50)

if __name__ == "__main__":
    main() 