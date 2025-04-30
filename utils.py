# utils.py
chars = ""
string_to_int = {}
int_to_string = {}

def init_vocab(text_file):
    global chars, string_to_int, int_to_string
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [string_to_int[c] for c in s]

def decode(l):
    return ''.join([int_to_string[i] for i in l])