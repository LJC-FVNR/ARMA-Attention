import torch
import torch.nn as nn

from math import gcd
from functools import reduce

def lcm(a, b):
    """Compute the least common multiple of a and b."""
    return a * b // gcd(a, b)

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def generate_first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

def generate_moduli(max_modulus, primes_only):
    if primes_only:
        return generate_first_n_primes(max_modulus)
    else:
        return list(range(2, max_modulus + 2))

def generate_modulus_encoding(n, moduli):
    return [n % m for m in moduli]

def generate_multi_hot_encoding(max_len, moduli, start_idx=1):
    total_moduli_sum = sum(moduli)
    multi_hot_tensor = torch.zeros((max_len, total_moduli_sum), dtype=torch.long)
    for n in range(max_len):
        encoding = generate_modulus_encoding(n+start_idx, moduli)
        current_idx = 0
        for mod_idx, mod in enumerate(moduli):
            multi_hot_tensor[n, current_idx + encoding[mod_idx]] = 1
            current_idx += mod
    return multi_hot_tensor

def calculate_max_non_repeating_integers(max_modulus=16, primes_only=False, start_idx=1):
    """Calculate the maximum number of non-repeating integers that can be represented with the given moduli."""
    moduli = generate_moduli(max_modulus, primes_only)
    # Compute the least common multiple of all moduli
    lcm_value = reduce(lcm, moduli) - start_idx
    print(f"Maximum number of non-repeating integers that can be represented with {max_modulus} moduli (primes_only={primes_only}, start_index={start_idx}): {lcm_value}")
    return lcm_value

class PositionEncodingModule(nn.Module):
    def __init__(self, max_modulus, max_len, target_dim, primes_only=True):
        super(PositionEncodingModule, self).__init__()
        self.moduli = generate_moduli(max_modulus, primes_only)
        self.max_len = max_len
        self.target_dim = target_dim
        self.total_moduli_sum = sum(self.moduli)
        multi_hot_encodings = generate_multi_hot_encoding(max_len, self.moduli, start_idx=1).float()
        self.register_buffer('multi_hot_encodings', multi_hot_encodings)
        self.processor = nn.Linear(self.target_dim, self.target_dim)
        self.linear = nn.Linear(self.total_moduli_sum, target_dim)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.processor.weight, mean=0.0, std=0.02)

        print(f'Current Mod Basis: {self.moduli}, Current Mod Multi-hot Dimension: {self.total_moduli_sum}')
        calculate_max_non_repeating_integers(max_modulus, primes_only)
    
    def forward(self, current_length):
        multi_hot_tensor = self.multi_hot_encodings[:current_length].unsqueeze(0)
        projected_tensor = self.linear(multi_hot_tensor) / multi_hot_tensor.sum(dim=-1, keepdim=True)
        projected_tensor = self.processor(projected_tensor)
        return projected_tensor