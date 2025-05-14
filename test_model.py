#!/usr/bin/env python
# filepath: /home/jack/GPT-from-scratch/test_model.py
import torch
import argparse
import os
from pathlib import Path
from GPT import GPTLanguageModel, encode, decode

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test the optimized GPT language model")
parser.add_argument("--model-path", type=str, default="checkpoints", help="Path to model checkpoint or directory")
parser.add_argument("--prompt", type=str, default="What is the essence of math?\n", help="Prompt for text generation")
parser.add_argument("--max-tokens", type=int, default=200, help="Maximum number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more focused)")
parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling parameter (0 to disable)")
parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
args = parser.parse_args()

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 
                     ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

def load_model(model_path):
    """Load a trained GPT model from checkpoint"""
    # Check if path is a directory (checkpoint folder)
    if os.path.isdir(model_path):
        print(f"Looking for latest checkpoint in {model_path}...")
        checkpoint_files = list(Path(model_path).glob("checkpoint_*.pt"))
        best_model = Path(model_path) / "best_model.pt"
        
        if os.path.exists(best_model):
            model_path = str(best_model)
            print(f"Found best model: {model_path}")
        elif checkpoint_files:
            # Find the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
            model_path = str(latest_checkpoint)
            print(f"Found latest checkpoint: {model_path}")
        else:
            print(f"No checkpoints found in {model_path}")
            # Try to find model-01.pth in the current directory
            if os.path.exists("model-01.pth"):
                model_path = "model-01.pth"
                print(f"Falling back to: {model_path}")
            else:
                raise FileNotFoundError(f"No model files found in {model_path} and no model-01.pth found")
    # Check if the specified file exists
    elif not os.path.exists(model_path):
        # Try alternate extensions
        alternate_paths = [
            model_path.replace(".pt", ".pth"),
            model_path.replace(".pth", ".pt"),
            model_path.replace(".pkl", ".pt"),
            "model-01.pth"
        ]
        
        for alt_path in alternate_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternate model file: {model_path}")
                break
        else:
            raise FileNotFoundError(f"Model file not found: {model_path} (and no alternatives found)")
    
    print(f"Loading model from {model_path}...")
    
    # Load vocabulary
    with open('training_set/textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = GPTLanguageModel(vocab_size).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint, iteration: {checkpoint.get('iteration', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded from state dict")
    
    return model

def generate_text(model, prompt, max_new_tokens=100, temperature=0.8, top_k=40, top_p=0.9):
    """Generate text using the trained model"""
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")
    
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        output = model.generate(
            context, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p
        )
    
    generated_text = decode(output[0].tolist())
    return generated_text

def main():
    # Load model
    try:
        model = load_model(args.model_path)
        
        # Generate text
        generated_text = generate_text(
            model, 
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Output the generated text
        print("\n" + "="*50)
        print("GENERATED TEXT:")
        print("="*50)
        print(generated_text)
        print("="*50)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
