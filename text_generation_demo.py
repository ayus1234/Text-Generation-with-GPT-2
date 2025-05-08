import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Define prompt
    prompt = "Once upon a time"
    
    print("\n" + "="*50)
    print("DECODING METHODS COMPARISON")
    print("="*50)
    
    # 1. Greedy Search
    print("\n1. Greedy Search")
    print("-"*30)
    greedy_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        decoding_strategy="greedy",
        max_length=50
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {prompt}{greedy_texts[0]}")
    
    # 2. Beam Search
    print("\n2. Beam Search")
    print("-"*30)
    beam_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        decoding_strategy="beam",
        num_beams=5,
        max_length=50
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {prompt}{beam_texts[0]}")
    
    # 3. Beam Search with No-Repeat N-Gram
    print("\n3. Beam Search with No-Repeat N-Gram")
    print("-"*30)
    beam_no_repeat_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        decoding_strategy="beam",
        num_beams=5,
        no_repeat_ngram_size=2,
        max_length=50
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {prompt}{beam_no_repeat_texts[0]}")
    
    # 4. Sampling
    print("\n4. Sampling")
    print("-"*30)
    sample_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        decoding_strategy="sample",
        temperature=1.0,
        max_length=50
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {prompt}{sample_texts[0]}")
    
    # 5. Top-K Sampling
    print("\n5. Top-K Sampling")
    print("-"*30)
    top_k_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        decoding_strategy="top_k",
        top_k=50,
        temperature=0.7,
        max_length=50
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {prompt}{top_k_texts[0]}")
    
    # 6. Top-p (Nucleus) Sampling
    print("\n6. Top-p (Nucleus) Sampling")
    print("-"*30)
    top_p_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        decoding_strategy="top_p",
        top_p=0.92,
        temperature=0.7,
        max_length=50
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {prompt}{top_p_texts[0]}")
    
    # 7. Multiple Sequences with Top-p
    print("\n7. Multiple Sequences with Top-p")
    print("-"*30)
    multi_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        decoding_strategy="top_p",
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=3,
        max_length=50
    )
    print(f"Prompt: {prompt}")
    print("Generated multiple texts:")
    for i, text in enumerate(multi_texts):
        print(f"\n--- Sequence {i+1} ---")
        print(f"{prompt}{text}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 