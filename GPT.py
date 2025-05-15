import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap
import random
import pickle
import argparse
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser(description='GPT Language Model Training')
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
parser.add_argument('--max-iters', type=int, default=5000, help='Maximum iterations for training')
parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')
args = parser.parse_args()

# Choose the appropriate device (CUDA, MPS, or CPU)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Hyperparameters
block_size = 256  # Increased context window
batch_size = 64   # Larger batch size for better gradient estimates
max_iters = args.max_iters
learning_rate = 6e-4  # Slightly higher learning rate with warmup
eval_iters = 100
n_embd = 768  # Larger embedding dimension
n_head = 12   # More attention heads
n_layer = 8   # More transformer layers
dropout = 0.1  # Reduced dropout for better convergence

# Vocabulary setup
with open('training_set/llm.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# Data loading with safety checks
def get_random_chunk(split):
    filename = f'training_set/{split}_split.txt'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Split file {filename} not found")
    
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            min_size = block_size * batch_size + 1
            if file_size < min_size:
                raise ValueError(f"File {filename} too small. Needs at least {min_size} bytes")
            
            max_start = max(0, file_size - block_size * batch_size)
            start_pos = random.randint(0, max_start)
            mm.seek(start_pos)
            
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
            # Ensure sufficient length
            while len(data) < block_size + 1:
                start_pos = random.randint(0, max(0, file_size - block_size * batch_size))
                mm.seek(start_pos)
                block = mm.read(block_size * batch_size - 1)
                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
                data = torch.tensor(encode(decoded_block), dtype=torch.long)
                
            return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

""" one head of self-attention with optimized attention calculation """
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # Optimized scaled dot-product attention with better numerical stability
        # Using a more efficient attention implementation that leverages PyTorch's optimizations
        scale = 1.0 / (self.head_size ** 0.5)
        q = q * scale  # Pre-scale query to avoid numerical instability
        
        # Apply causal mask and compute attention weights
        attn = torch.bmm(q, k.transpose(-2, -1))  # (B, T, T)
        mask = self.tril[:T, :T] == 0
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)  # (B, T, T)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # (B, T, head_size)
        return out

""" multiple heads of self-attention in parallel """
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate outputs of all heads
        out = self.dropout(self.proj(out))
        return out

""" a simple feed-forward network with GELU activation """
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(), # use GELU instead of ReLU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

""" Transformer block: attention + feed-forward with residual connections """
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x

""" GPT model with improved structure """
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        """
        Generate text using the model with enhanced sampling strategies
        
        Args:
            index: starting tokens
            max_new_tokens: number of tokens to generate
            temperature: controls randomness (lower = more deterministic)
            top_k: sample from top k most likely tokens (if specified)
            top_p: sample from tokens comprising top p probability mass (nucleus sampling)
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            index_cond = index[:, -block_size:]
            
            # Forward pass
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :] # only take the last token's predictions
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence and continue
            index = torch.cat((index, index_next), dim=1)
        return index

# Create checkpoints directory
os.makedirs(args.checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, iter_num, best_val_loss, is_best=False):
    """Save a checkpoint of the model and optimizer state"""
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{iter_num:06d}.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iter_num,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # If this is the best model, save a separate copy
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")
        
def load_latest_checkpoint():
    """Load the latest checkpoint from the checkpoint directory"""
    checkpoint_files = list(Path(args.checkpoint_dir).glob("checkpoint_*.pt"))
    if not checkpoint_files:
        print("No checkpoints found.")
        return None
    
    # Find the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint
    checkpoint = torch.load(str(latest_checkpoint), map_location=device)
    return checkpoint

# Only run training code if this file is run directly (not imported)
if __name__ == "__main__":
    # Training setup
    model = GPTLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    
    # Resume training if specified
    start_iter = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = load_latest_checkpoint()
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iteration']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed training from iteration {start_iter}, best val loss: {best_val_loss:.3f}")
    
    # Learning rate scheduler with linear warmup and cosine decay
    warmup_iters = 1000  # Number of iterations for warmup
    def get_lr(iter):
        # Linear warmup for warmup_iters steps
        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters
        # Cosine decay after warmup_iters steps
        decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
        return learning_rate * 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.14159)).item())
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            try:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    _, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            except Exception as e:
                print(f"Error evaluating {split}: {str(e)}")
                out[split] = float('nan')
        model.train()
        return out
    
    # Training loop with proper split handling
    for iter in range(start_iter, max_iters):
        # Evaluation
        if iter % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {iter}: Train Loss {losses['train']:.3f}, Val Loss {losses['val']:.3f}")
            is_best = losses['val'] < best_val_loss
            if is_best:
                best_val_loss = losses['val']
                print(f"Saving best model with val loss: {losses['val']:.3f}")
            save_checkpoint(model, optimizer, iter, best_val_loss, is_best=is_best)
        
        # Dynamic learning rate adjustment
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training
        try:
            xb, yb = get_batch('train')  # Explicit training split
            _, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Log training progress every 100 steps
            if iter % 100 == 0:
                print(f"Iteration {iter}: LR = {lr:.6f}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Training error at step {iter}: {str(e)}")
            break
    
    # Final save and generation
    torch.save(model.state_dict(), 'model-01.pth')
    print("Model saved")
    
    prompt = "What is the essence of math?\n"
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = decode(model.generate(context, max_new_tokens=100)[0].tolist())
    print(generated)