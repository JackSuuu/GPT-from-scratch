import torch
from GPT import GPTLanguageModel  # Your model class
from utils import encode, decode   # Your tokenization functions

# --- Config (MUST match training) ---
device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
block_size = 128
n_embd = 512
n_head = 8
n_layer = 6

# --- Load Model ---
def load_model(model_path):
    # 1. Recreate model architecture
    model = GPTLanguageModel(vocab_size=len(chars))  # chars loaded from utils.py
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- Generation Only ---
def generate_text(model, prompt, max_tokens=100):
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    return decode(generated)

if __name__ == "__main__":
    # Initialize vocabulary (shared utils.py)
    from utils import init_vocab
    init_vocab('training_set/textbook.txt')  # Same as training

    # Load trained model
    model = load_model('model-01.pth')
    
    # Example usage
    prompt = "What is probability distribution?"
    print(generate_text(model, prompt))