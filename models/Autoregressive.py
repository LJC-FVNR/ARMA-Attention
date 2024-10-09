import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from types import SimpleNamespace
from layers.MEGA import MovingAverageGatedAttention

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        From: https://github.com/meta-llama/llama/blob/main/llama/model.py
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @classmethod
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight

NORM = RMSNorm # nn.LayerNorm (RMS norm provides similar behavior, with lower computational costs)

# Activation functions for Q_MA and K_MA
def ma_q_activation(x):
    x = - x / math.sqrt(x.shape[-1])
    x = - F.leaky_relu(x, negative_slope=0.02)
    return x

def ma_k_activation(x, k=0.02):
    x = x / math.sqrt(x.shape[-1])
    x = 1 / (1 + torch.exp(-x * k))
    return x

def ma_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, return_weight=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    query = ma_q_activation(query)
    key = ma_k_activation(torch.dropout(key, dropout_p, train=True))
    attn_weight = query @ key.transpose(-2, -1)
    attn_weight = attn_weight.tril(diagonal=0)
    attn_weight = attn_weight
    if return_weight:
        return attn_weight
    return attn_weight @ value

def generate_attn_weight_from_qk(q, k, scale=True, softmax=True, diagonal=0):
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale else 1
    attn_bias = torch.zeros(L, S, dtype=q.dtype)
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=diagonal)
    attn_bias.masked_fill_(temp_mask.logical_not(), -1000)
    attn_bias.to(q.dtype)
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    if softmax:
        attn_weight += attn_bias
    else:
        attn_weight *= temp_mask.float()
    attn_weight = torch.softmax(attn_weight, dim=-1) if softmax else attn_weight
    if diagonal == -1:
        attn_weight[:, :, 0, :] = 0
    return attn_weight


# CausalSelfAttention - FullAttention - Standard Softmax Attention
class CausalSelfAttention(nn.Module):
    ### From: https://github.com/karpathy/nanoGPT/blob/master/model.py
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = torch.cumsum(att, dim=2)
            att = torch.cumsum(att, dim=3)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# CausalSelfAttention - FullAttentionARMA - Standard Softmax Attention + ARMA
class CausalSelfAttentionARMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias) # for q and k
        self.k2 = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        #self.ma_activation = nn.ReLU()
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.ma_activation = nn.ReLU()
        self.dropout_ma = nn.Dropout(config.ma_dropout)
        self.ma_dropout_rate = config.ma_dropout

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k  = self.c_attn(x).split(self.n_embd, dim=2)
        v = x
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        e = v[:, :, 1:, :] - y[:, :, :-1, :]
        k2 = self.k2(x[:, :-1]).view(B, T-1, self.n_head, -1).transpose(1, 2) # (B, nh, T, hs)
        q2 = q[:, :, 1:, :]

        y2 = ma_scaled_dot_product_attention(q2, k2, e, attn_mask=None, dropout_p=self.ma_dropout_rate if self.training else 0, is_causal=True)
        y2 = torch.cat([torch.zeros_like(y2[:, :, :1, :]), y2], dim=2)
        y = self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))) + self.dropout_ma(self.c_proj(y2.transpose(1, 2).contiguous().view(B, T, C))) # re-assemble all head outputs side by side
        return y

    def calculate_arma_weights(self, x, channel=None, channel_average=False, normalize=True, output_sequence=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        weights_ar = generate_attn_weight_from_qk(q, k, scale=True, softmax=normalize, diagonal=0)
        if channel is not None:
            weights_ar = weights_ar[:, channel]
            
        k2 = self.k2(x[:, :-1]).view(B, T-1, self.n_head, -1).transpose(1, 2) # (B, nh, T, hs)
        q2 = q[:, :, 1:, :]

        Beta_output = ma_scaled_dot_product_attention(q2, k2, None, attn_mask=None, dropout_p=self.ma_dropout_rate if self.training else 0, is_causal=True, return_weight=True)
        Beta = torch.zeros(Beta_output.shape[0], Beta_output.shape[1], T, T, device=x.device)
        Beta[:, :, 1:, 0:-1] = Beta_output
        if channel is not None:
            Beta = Beta[:, channel]

        if channel is not None:
            I = torch.eye(Beta.shape[-1], device=x.device).unsqueeze(0).expand(B, -1, -1)
            Beta_inverse = torch.inverse(I - Beta) # torch.linalg.pinv(I - Beta)
            weights_ma = Beta @ Beta_inverse   # Omega
        else:
            I = torch.eye(Beta.shape[-1], device=x.device).unsqueeze(0).unsqueeze(0).expand(B, Beta.shape[1], -1, -1)
            Beta_inverse = torch.inverse(I - Beta) # torch.linalg.pinv(I - Beta)
            weights_ma = torch.einsum('bdli,bdij->bdlj', Beta, Beta_inverse) # Beta @ Beta_inverse   # Omega
        if channel is None and channel_average:
            weights_ar = weights_ar.mean(dim=1)
            weights_ma = weights_ma.mean(dim=1)
            Beta = Beta.mean(dim=1)

        if not output_sequence:
            return weights_ar, weights_ma, Beta
        else:
            v = x
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            e = v[:, :, 1:, :] - y[:, :, :-1, :]
            y2 = ma_scaled_dot_product_attention(q2, k2, e, attn_mask=None, dropout_p=self.ma_dropout_rate if self.training else 0, is_causal=True)
            y2 = torch.cat([torch.zeros_like(y2[:, :, :1, :]), y2], dim=2)
            yf = self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))) + self.dropout_ma(self.c_proj(y2.transpose(1, 2).contiguous().view(B, T, C)))
            return weights_ar, weights_ma, Beta, y.transpose(1, 2).contiguous().view(B, T, C), yf

# MaskedLinear - Fixed Attention
class MaskedLinear(nn.Module):
    def __init__(self, config, generative_dependency=False):
        super(MaskedLinear, self).__init__()
        self.in_features = config.max_len
        self.out_features = config.max_len
        self.n_heads = config.n_head
        D = config.n_embd
        self.d = D // self.n_heads
        self.weight = nn.Parameter(torch.zeros(self.n_heads, self.out_features, self.in_features))
        self.register_buffer('mask', torch.tril(torch.ones(self.out_features, self.in_features)))
        self.register_buffer('norm_factor', torch.cumsum(self.mask, dim=0) + 1e-7)
        self.activation = nn.Softmax(dim=-1)
        self.c_proj = nn.Linear(D, D, bias=config.bias)
        self.generative_dependency = generative_dependency
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        # input: B L D
        B, L, D = input.shape
        input = input.reshape(B, L, self.n_heads, self.d) # B L H d
        masked_weight = self.weight * self.mask
        if self.generative_dependency:
            masked_weight = torch.cumsum(masked_weight, dim=1)
        normalized_weight = self.dropout(self.activation(masked_weight))
        linear_res = torch.einsum('blhd,hil->bihd', input, normalized_weight)
        linear_res = linear_res.reshape(B, L, D)
        linear_res = self.dropout(self.c_proj(linear_res))
        return linear_res

# MaskedLinearARMA - Fixed Attention + ARMA
class MaskedLinearARMA(nn.Module):
    def __init__(self, config, generative_dependency=False):
        super(MaskedLinearARMA, self).__init__()
        self.in_features = config.max_len
        self.out_features = config.max_len
        self.n_heads = config.n_head
        D = config.n_embd
        self.d = D // self.n_heads
        self.weight = nn.Parameter(torch.zeros(self.n_heads, self.out_features, self.in_features))
        self.register_buffer('mask', torch.tril(torch.ones(self.out_features, self.in_features)))
        self.register_buffer('norm_factor', torch.cumsum(self.mask, dim=0) + 1e-7)
        self.weight_ma_q = nn.Parameter(torch.zeros(self.n_heads, self.out_features, D))
        self.weight_ma_k = nn.Parameter(torch.zeros(self.n_heads, D, self.in_features))
        self.register_buffer('mask_ma', torch.tril(torch.ones(self.out_features, self.in_features), diagonal=-1))
        self.activation = nn.Softmax(dim=-1)
        self.activation_ma = nn.Softmax(dim=-1)
        self.c_proj = nn.Linear(D, D, bias=config.bias)
        self.generative_dependency = generative_dependency
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_ma = nn.Dropout(config.ma_dropout)

    def forward(self, input):
        # input: B L D
        B, L, D = input.shape
        input = input.reshape(B, L, self.n_heads, self.d) # B L H d
        masked_weight = self.weight * self.mask
        if self.generative_dependency:
            masked_weight = torch.cumsum(masked_weight, dim=1)
        normalized_weight = self.dropout(self.activation(masked_weight))
        linear_res = torch.einsum('blhd,hil->bihd', input, normalized_weight)
        weight_ma_q = ma_q_activation(self.weight_ma_q)
        weight_ma_k = self.dropout_ma(ma_k_activation(self.weight_ma_k))
        weight_ma = torch.einsum('hld,hdi->hli', weight_ma_q, weight_ma_k)
        masked_weight_ma = weight_ma * self.mask_ma
        if self.generative_dependency:
            masked_weight_ma = torch.cumsum(masked_weight_ma, dim=1)
        normalized_weight_ma = masked_weight_ma
        ma_input = torch.cat([torch.zeros_like(linear_res[:, :, :1, :]), input[:, :, 1:, :] - linear_res[:, :, :-1, :]], dim=2)
        linear_res2 = torch.einsum('blhd,hil->bihd', ma_input, normalized_weight_ma).reshape(B, L, D)
        linear_res = linear_res.reshape(B, L, D)
        linear_res = self.dropout(self.c_proj(linear_res)) + self.dropout_ma(self.c_proj(linear_res2))
        return linear_res

# LinearAttention - Linear Attention
class LinearAttention(nn.Module):
    def __init__(self, config):
        super(LinearAttention, self).__init__()
        self.D = config.n_embd
        self.activation = nn.SiLU()
        self.n_head = config.n_head

        assert self.D % self.n_head == 0
        self.d = self.D // self.n_head

        self.c_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.k1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.q1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.v1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, X):
        B, L, D = X.shape
        device = X.device
        
        Q = self.q1(X).reshape(B, L, self.n_head, -1)                        
        K = self.k1(self.dropout(X)).reshape(B, L, self.n_head, -1).unsqueeze(-1)     # B L H d 1
        V = self.v1(X).reshape(B, L, self.n_head, -1).unsqueeze(-1).permute(0, 1, 2, 4, 3)   # B L H 1 d
        W = torch.einsum('blhdk,blhke->blhde', K, V)         # B L H d d
        W = torch.cumsum(W, dim=1)
        X = torch.einsum('blhd,blhde->blhe', Q, W)
        X = X.reshape(B, L, D)
        X = self.dropout(self.c_proj(X))
        return X

# LinearAttentionARMA - Linear Attention + ARMA
class LinearAttentionARMA(nn.Module):
    def __init__(self, config):
        super(LinearAttentionARMA, self).__init__()
        self.D = config.n_embd
        self.activation = nn.SiLU()
        self.n_head = config.n_head

        assert self.D % self.n_head == 0
        self.d = self.D // self.n_head

        self.c_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.k1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.q1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.k2 = nn.Linear(self.D, self.D, bias=config.bias)
        self.q2_activation = ma_q_activation
        self.k2_activation = ma_k_activation
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_ma = nn.Dropout(config.ma_dropout)
    
    def forward(self, X):
        B, L, D = X.shape
        device = X.device
        Q = self.q1(X).reshape(B, L, self.n_head, -1)                        
        K = self.k1(self.dropout(X)).reshape(B, L, self.n_head, -1).unsqueeze(-1)     # B L H d 1
        V = X.reshape(B, L, self.n_head, -1).unsqueeze(-1).permute(0, 1, 2, 4, 3)   # B L H 1 d
        
        k2 = self.dropout_ma(self.k2_activation(self.k2(X[:, :-1]))).reshape(B, L-1, self.n_head, -1).unsqueeze(-1)
        q2 = self.q2_activation(Q[:, :-1])
        
        W = torch.einsum('blhdk,blhke->blhde', K, V)         # B L H d d
        W = torch.cumsum(W, dim=1)
        X = torch.einsum('blhd,blhde->blhe', Q, W)           # B L H d

        Y = X.unsqueeze(-2)                                                                 # B L H 1 d
        E = V[:, 1:] - Y[:, :-1]
        WE = torch.einsum('blhdk,blhke->blhde', k2, E)         # B L H d d
        WE = torch.cumsum(WE, dim=1)
        XE = torch.einsum('blhd,blhde->blhe', q2, WE)
        XE = torch.cat([torch.zeros_like(XE[:, [0]]), XE], dim=1)

        X = self.dropout(X).reshape(B, L, D)
        XE = self.dropout_ma(X).reshape(B, L, D)
        
        X = self.c_proj(X + XE)
        return X

def weighted_cumsum(input_tensor, weights, dim):
    assert input_tensor.shape[dim] == weights.shape[dim], "Input tensor and weights must have the same shape along the specified dimension"
    cumprod_weights = calculate_weighted_cumsum_weight(weights, dim)
    result = apply_weighted_cumsum_weight(input_tensor, cumprod_weights, dim)
    return result

def calculate_weighted_cumsum_weight(weights, dim):
    weights = torch.clamp(weights, min=1e-6)
    log_weights = torch.log(weights)
    log_cumprod_weights = torch.cumsum(log_weights, dim=dim)
    log_cumprod_weights = torch.clamp(log_cumprod_weights, max=30, min=-30)
    cumprod_weights = torch.exp(log_cumprod_weights) + 1e-6
    return cumprod_weights

def apply_weighted_cumsum_weight(input_tensor, cumprod_weights, dim):
    result = torch.cumsum(input_tensor / cumprod_weights, dim=dim) * cumprod_weights
    return result

# GatedLinearAttention - Gated Linear Attention
class GatedLinearAttention(nn.Module):
    def __init__(self, config):
        super(GatedLinearAttention, self).__init__()
        self.D = config.n_embd
        self.activation = nn.SiLU()
        self.n_head = config.n_head

        assert self.D % self.n_head == 0
        self.d = self.D // self.n_head

        self.c_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.k1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.q1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.v1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.gw = nn.Linear(self.d, 1, bias=config.bias)
        self.decay = config.decay
        if self.decay:
            self.sw = nn.Linear(self.d, 1, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, X):
        B, L, D = X.shape
        Q = self.q1(X).reshape(B, L, self.n_head, -1)                        
        K = self.k1(self.dropout(X)).reshape(B, L, self.n_head, -1).unsqueeze(-1)     # B L H d 1
        V = self.v1(X).reshape(B, L, self.n_head, -1).unsqueeze(-1).permute(0, 1, 2, 4, 3)   # B L H 1 d
        W = torch.einsum('blhdk,blhke->blhde', K, V)         # B L H d d
        G = F.sigmoid(self.gw(X.view(B, L, self.n_head, -1)))   # B L H 1
        G = calculate_weighted_cumsum_weight(G, dim=1).unsqueeze(-1).expand(-1, -1, -1, self.d, self.d)   # B L H 1 1 -> B L H d d
        if self.decay:
            R = F.silu(self.sw(K[:, :, :, :, 0])).unsqueeze(-1).expand(-1, -1, -1, self.d, self.d)   # B L H 1
        else:
            R = 1
        W = apply_weighted_cumsum_weight(W*R, G, dim=1)
        X = torch.einsum('blhd,blhde->blhe', Q, W)
        X = X.reshape(B, L, D)
        X = self.dropout(self.c_proj(X))
        return X

# GatedLinearAttentionARMA - Gated Linear Attention + ARMA
class GatedLinearAttentionARMA(nn.Module):
    def __init__(self, config):
        super(GatedLinearAttentionARMA, self).__init__()
        self.D = config.n_embd
        self.activation = nn.SiLU()
        self.n_head = config.n_head

        assert self.D % self.n_head == 0
        self.d = self.D // self.n_head

        self.c_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.k1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.q1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.k2 = nn.Linear(self.D, self.D, bias=config.bias)
        self.gw = nn.Linear(self.d, 1, bias=config.bias)
        self.decay = config.decay
        if self.decay:
            self.sw = nn.Linear(self.d, 1, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_ma = nn.Dropout(config.ma_dropout)
    
    def forward(self, X):
        B, L, D = X.shape
        device = X.device
        
        Q = self.q1(X).reshape(B, L, self.n_head, -1)                        
        K = self.k1(self.dropout(X)).reshape(B, L, self.n_head, -1).unsqueeze(-1)     # B L H d 1
        V = X.reshape(B, L, self.n_head, -1).unsqueeze(-1).permute(0, 1, 2, 4, 3)   # B L H 1 d
        W = torch.einsum('blhdk,blhke->blhde', K, V)         # B L H d d
        G = F.sigmoid(self.gw(X.view(B, L, self.n_head, -1)))   # B L H 1
        G = calculate_weighted_cumsum_weight(G, dim=1).unsqueeze(-1).expand(-1, -1, -1, self.d, self.d)   # B L H 1 1 -> B L H d d
        if self.decay:
            R = F.silu(self.sw(K[:, :, :, :, 0])).unsqueeze(-1).expand(-1, -1, -1, self.d, self.d)   # B L H 1
        else:
            R = 1
        W = apply_weighted_cumsum_weight(W*R, G, dim=1)
        O1 = torch.einsum('blhd,blhde->blhe', Q, W)
        
        E = V[:, 1:, :] - O1.unsqueeze(-2)[:, :-1, :]
        Q = ma_q_activation(Q[:, :-1])
        K = self.dropout_ma(ma_k_activation(self.k2(X[:, :-1]))).reshape(B, L-1, self.n_head, -1).unsqueeze(-1)
        W = torch.einsum('blhdk,blhke->blhde', K, E)         # B L H d d
        W = torch.cumsum(W, dim=1)
        O2 = torch.einsum('blhd,blhde->blhe', Q, W)
        O2 = torch.cat([torch.zeros_like(O2[:, :1]), O2], dim=1)

        O1 = self.dropout(O1.reshape(B, L, D))
        O2 = self.dropout_ma(O2.reshape(B, L, D))
        X = self.c_proj(O1 + O2)
        return X

# TwoStageSelfgatingRNN - Element-wise Linear Attention (AFT)
class TwoStageSelfgatingRNN(nn.Module):
    def __init__(self, config):
        super(TwoStageSelfgatingRNN, self).__init__()
        self.D = config.n_embd
        self.activation_Q = nn.Sigmoid()

        self.c_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.q1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.k1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.v1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        B, L, D = X.shape
        device = X.device         
        V = self.v1(X)
        K = self.k1(X)      
        K = self.dropout(torch.exp(K))
        K_factor = K.cumsum(dim=1) + 1e-6
        H = K*V
        S = H.cumsum(dim=1) / K_factor
        Q = self.activation_Q(self.q1(X))
        O = S * Q
        X = self.dropout(self.c_proj(O))
        return X

# TwoStageSelfgatingRNNARMA - Element-wise Linear Attention (AFT) + ARMA
class TwoStageSelfgatingRNNARMA(nn.Module):
    def __init__(self, config):
        super(TwoStageSelfgatingRNNARMA, self).__init__()
        self.D = config.n_embd
        self.activation_Q = nn.Sigmoid()
        self.activation_MA = nn.Sigmoid()

        self.c_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.q1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.k1 = nn.Linear(self.D, self.D, bias=config.bias)

        #self.q2 = nn.Linear(self.D, self.D, bias=config.bias)
        self.k2 = nn.Linear(self.D, self.D, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)
        self.dropout_ma = nn.Dropout(config.ma_dropout)

    def forward(self, X):
        B, L, D = X.shape
        device = X.device         
        V = X 
        K = self.k1(X)           
        K = self.dropout(torch.exp(K))
        K_factor = K.cumsum(dim=1) + 1e-6
        H = K*V
        S = H.cumsum(dim=1) / K_factor
        Q = self.activation_Q(self.q1(X))
        O = S * Q
        
        E = torch.cat([torch.zeros_like(O[:, :1, :]), V[:, 1:, :] - O[:, :-1, :]], dim=1)
        QK2_input = torch.cat([torch.zeros_like(X[:, :1, :]), X[:, :-1, :]], dim=1)
        K = self.k2(QK2_input)
        K = self.dropout_ma(ma_k_activation(K))
        H = K*E
        S = H.cumsum(dim=1)
        Q = ma_q_activation(Q)
        Q = torch.cat([torch.zeros_like(Q[:, :1, :]), Q[:, :-1, :]], dim=1)
        X = self.c_proj(self.dropout(O) + self.dropout_ma(S * Q))
        return X

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # , bias=config.bias
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layer = config.n_layer
        self.n_layer = n_layer
        self.predictor = config.predictor
        if self.predictor == 'FullAttention':
            self.attn = nn.ModuleList([CausalSelfAttention(config) for i in range(n_layer)])
        elif self.predictor == 'FullAttentionARMA' or self.predictor == 'Default':
            self.attn = nn.ModuleList([CausalSelfAttentionARMA(config) for i in range(n_layer)])
        elif self.predictor == 'MaskedLinear':
            self.attn = nn.ModuleList([MaskedLinear(config) for i in range(n_layer)])
        elif self.predictor == 'MaskedLinearARMA':
            self.attn = nn.ModuleList([MaskedLinearARMA(config) for i in range(n_layer)])
        elif self.predictor == 'LinearAttention':
            self.attn = nn.ModuleList([LinearAttention(config) for i in range(n_layer)])
        elif self.predictor == 'LinearAttentionARMA':
            self.attn = nn.ModuleList([LinearAttentionARMA(config) for i in range(n_layer)])
        elif self.predictor == 'GatedLinearAttention':
            self.attn = nn.ModuleList([GatedLinearAttention(config) for i in range(n_layer)])
        elif self.predictor == 'GatedLinearAttentionARMA':
            self.attn = nn.ModuleList([GatedLinearAttentionARMA(config) for i in range(n_layer)])
        elif self.predictor == 'TwoStageSelfgatingRNN':
            self.attn = nn.ModuleList([TwoStageSelfgatingRNN(config) for i in range(n_layer)])
        elif self.predictor == 'TwoStageSelfgatingRNNARMA':
            self.attn = nn.ModuleList([TwoStageSelfgatingRNNARMA(config) for i in range(n_layer)])
        elif self.predictor == 'MovingAverageGatedAttention':
            self.attn = nn.ModuleList([MovingAverageGatedAttention(embed_dim=config.n_embd, zdim=config.n_embd//4, hdim=2*config.n_embd, ndim=config.n_embd//32) for i in range(n_layer)])
        self.pns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.lns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for i in range(n_layer)])

    def forward(self, x):
        for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
            x = x + attn(pn(x))
            x = x + mlp(ln(x))
        return x

class SegmentTokenizer(nn.Module):
    def __init__(self, s: int, d_model: int):
        super(SegmentTokenizer, self).__init__()
        self.s = s
        self.tokenizer = nn.Linear(s, d_model)
        self.tokenizer_output = nn.Linear(d_model, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B C L -> # B C N d
        B, C, L = x.shape
        pad_len = (self.s - (L % self.s)) % self.s
        x_padded = F.pad(x, (pad_len, 0), 'constant', 0)
        L_padded = L + pad_len
        N = L_padded // self.s
        x_segmented = x_padded.view(B, C, N, self.s)
        return x_segmented

    def tokenize(self, x_segmented):
        return self.tokenizer(x_segmented)

    def inverse_tokenize(self, x):
        # B C N d -> B C N*s
        B, C, N, d = x.shape
        #x = torch.einsum('bcnd,ds->bcns', x - self.tokenizer.bias, self.tokenizer.weight)
        x = self.tokenizer_output(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        pad_len = (self.pred_len - (self.seq_len % self.pred_len)) % self.pred_len
        self.max_len = (self.seq_len + pad_len) // self.pred_len
        self.d_model = int(np.sqrt(configs.enc_in)) * 16 if not configs.d_model else configs.d_model
        self.tokenizer = SegmentTokenizer(s=self.pred_len, d_model=self.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.pre_norm = NORM(self.d_model)
        self.final_norm = NORM(self.d_model)
        transformer_config = SimpleNamespace(n_layer=configs.e_layers, n_embd=self.d_model, n_head=configs.n_heads, dropout=configs.dropout, 
                                             bias=True, max_len=self.max_len, predictor=configs.predictor, decay=configs.decay, ma_dropout=configs.ma_dropout)
        self.transformer = Transformer(transformer_config)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.channel_emb = nn.Parameter(0.02*torch.randn(1, configs.enc_in, self.max_len, self.d_model))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * configs.e_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, *args, **kwargs):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # B C L

        # RevIN: preprocessing
        mean = x[:, :, -self.pred_len:].mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-6
        x = (x - mean) / std
        x_segmented = self.tokenizer(x)   # B C N s

        # Input Tokenization
        x = self.tokenizer.tokenize(x_segmented)   # B C N d
        B, C, N, d = x.shape
        pos = torch.arange(0, N, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos) + self.channel_emb[:, :, 0:N, :]
        x = self.dropout(x) + pos_emb  # B C N d
        x = self.pre_norm(x)

        # AR/ARMA Transformer
        x = x.view(B*C, N, d)
        x = self.transformer(x)

        # Output Projection
        x = self.final_norm(x)     # B*C N d
        x = x.view(B, C, N, d)     # B C N d
        x = self.tokenizer.inverse_tokenize(x)  # B C N s

        # Next-step prediction loss
        if self.training:
            loss = F.mse_loss(x[:, :, 0:-1, :], x_segmented[:, :, 1:, :])

        # Next-step output
        x = x[:, :, -1, :]         # B C s

        # RevIN: inverse processing
        x = (x*std + mean).permute(0, 2, 1)                  # B s C

        if self.training:
            return x, loss
        else:
            return x