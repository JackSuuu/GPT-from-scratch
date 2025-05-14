#!/usr/bin/env python
# filepath: /home/jack/GPT-from-scratch/prepare_data.py
"""
Script to prepare training data for GPT model training.
This script splits the data into training and validation sets.
"""

import os
import argparse
import random
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare text data for GPT training')
    parser.add_argument('--input', type=str, default='training_set/textbook.txt',
                        help='Path to the input text file')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of validation data (between 0 and 1)')
    parser.add_argument('--output-dir', type=str, default='training_set',
                        help='Directory to save the split files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def create_data_splits(input_file, output_dir, val_ratio=0.1, seed=42):
    """
    Split the input text file into training and validation sets.
    
    Args:
        input_file (str): Path to the input text file
        output_dir (str): Directory to save the split files
        val_ratio (float): Ratio of validation data (between 0 and 1)
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input file
    print(f"Reading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    # Get the file size and calculate the split
    total_size = len(text)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    print(f"Total text size: {total_size} characters")
    print(f"Training set: {train_size} characters ({100 - val_ratio*100:.1f}%)")
    print(f"Validation set: {val_size} characters ({val_ratio*100:.1f}%)")
    
    # Create train and validation splits
    # We split by paragraphs to maintain context
    paragraphs = text.split('\n\n')
    random.shuffle(paragraphs)
    
    # Calculate how many paragraphs for validation
    total_paragraphs = len(paragraphs)
    val_paragraphs = int(total_paragraphs * val_ratio)
    
    # Create the splits
    val_text = '\n\n'.join(paragraphs[:val_paragraphs])
    train_text = '\n\n'.join(paragraphs[val_paragraphs:])
    
    # Save the splits
    train_path = os.path.join(output_dir, 'train_split.txt')
    val_path = os.path.join(output_dir, 'val_split.txt')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    print(f"Training data saved to {train_path}")
    print(f"Validation data saved to {val_path}")
    
    # Check if the files were created successfully
    train_file_size = os.path.getsize(train_path)
    val_file_size = os.path.getsize(val_path)
    
    print(f"Training file size: {train_file_size} bytes")
    print(f"Validation file size: {val_file_size} bytes")
    
    return train_path, val_path

def main():
    args = parse_arguments()
    
    print("Preparing data splits for GPT training...")
    create_data_splits(
        args.input, 
        args.output_dir,
        args.val_ratio,
        args.seed
    )
    print("Data preparation completed!")

if __name__ == "__main__":
    main()
