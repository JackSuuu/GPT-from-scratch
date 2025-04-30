import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap
import random
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description='This is a demonstration')
device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
print(device)

# Hyperparameters
block_size = 128
batch_size = 32
max_iters = 5000
learning_rate = 3e-4
eval_iters = 100
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.2

# Vocabulary setup
with open('training_set/textbook.txt', 'r', encoding='utf-8') as f:
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

""" one head of self-attention """
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
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
    
    def generate(self, index, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# Training setup
model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

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
best_val_loss = float('inf')
for iter in range(max_iters):
    # Evaluation
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.3f}, Val Loss {losses['val']:.3f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Training
    try:
        xb, yb = get_batch('train')  # Explicit training split
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
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