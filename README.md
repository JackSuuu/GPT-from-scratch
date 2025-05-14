# GPT-from-scratch

A from-scratch implementation of a GPT (Generative Pre-trained Transformer) model for learning purposes. This project provides a clean and minimal implementation to help understand the core concepts behind modern transformer-based language models.

## Parameters

```python
batch_size = 64   # Larger batch size for better gradient estimates
max_iters = args.max_iters
learning_rate = 6e-4  # Slightly higher learning rate with warmup
eval_iters = 100
n_embd = 768  # Larger embedding dimension
n_head = 12   # More attention heads
n_layer = 8   # More transformer layers
dropout = 0.1  # Reduced dropout for better convergence
```

This Transformer's parameter is approximately equivalent to GPT-2 Small (124M).

## Features

- Pure Python/PyTorch implementation
- Minimal dependencies
- Training and inference scripts
- Checkpoint saving/resuming
- Validation set evaluation
- Example data preparation script

## Requirements

- Python 3.7+
- PyTorch 1.8+
- tqdm (for progress bars)
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JackSuuu/GPT-from-scratch.git
cd GPT-from-scratch
```

2. Install the required packages:
```bash
pip install torch numpy tqdm
```

## Usage

### 1. Prepare Your Data

First, format your training data into a plain text file. Then run the preparation script:

```bash
python prepare_data.py --input training_set/textbook.txt --val-ratio 0.1
```

Arguments:
- `--input`: Path to your input text file
- `--val-ratio`: Ratio of data to use for validation (default: 0.1)
- `--output-dir`: Directory to save processed data (default: 'data')

This will create train/val splits and save them in the specified directory.

### 2. Train the Model

Train your GPT model with:

```bash
python GPT.py --checkpoint-dir checkpoints --max-iters 5000 --eval-interval 100
```

Key arguments:
- `--checkpoint-dir`: Directory to save model checkpoints (default: 'checkpoints')
- `--max-iters`: Maximum training iterations (default: 5000)
- `--eval-interval`: Evaluate on validation set every N steps (default: 100)
- `--batch-size`: Training batch size (default: 64)
- `--block-size`: Context length (default: 256)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--device`: 'cpu' or 'cuda' (default: try to use CUDA if available)

### 3. Resume Training

If training gets interrupted, resume from the latest checkpoint:

```bash
python GPT.py --resume --checkpoint-dir checkpoints --max-iters 5000
```

### 4. Test the Model

Generate text with your trained model:

```bash
python test_model.py --model-path checkpoints/best_model.pt --prompt "What is physics?" --max-tokens 200
```

Additional arguments:
- `--prompt`: Starting prompt text (default: "\n")
- `--max-new-tokens`: Maximum tokens to generate (default: 500)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-k`: Top-k sampling (default: 40)

## Implementation Details

The implementation includes:
- Multi-head self-attention
- Transformer blocks with residual connections
- Position-wise feedforward networks
- Layer normalization
- Learned position embeddings
- Byte-pair encoding tokenizer (basic implementation)
- Model checkpointing
- Training/validation loops

## Customization

You can easily modify:
- Model dimensions (embedding size, number of heads, layers)
- Training parameters (batch size, learning rate)
- Context length
- Tokenizer implementation

## License

This project is open source under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request for any improvements or bug fixes.