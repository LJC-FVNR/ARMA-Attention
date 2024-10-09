import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
        
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=128, depth=2, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=mlp_ratio*emb_size, batch_first=True, dropout=dropout, norm_first=True))
            
    def forward(self, x, mask=None, output_attention=False):
        attention_maps = []
        for layer in self.layers:
            if output_attention:
                x, attention = layer(x, src_mask=mask)
                attention_maps.append(attention)
            else:
                x = layer(x, src_mask=mask)
        if output_attention:
            return x, attention_maps
        return x
    
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        seq_len=configs.seq_len
        pred_len=configs.pred_len
        self.pred_len = pred_len
        depth=configs.e_layers
        heads=configs.n_heads
        mlp_ratio=configs.mlp_ratio
        d_model=configs.d_model
        emb_init=0.01
        n_channels=configs.enc_in
        dropout=configs.dropout
        self.predictor = 'Temporal-wise' # configs.predictor
        
        self.positional_embedding = nn.Parameter(emb_init*torch.randn(1, 8192, d_model))
        self.output_embedding = nn.Parameter(emb_init*torch.randn(1, pred_len, n_channels))
        
        if self.predictor == 'Temporal-wise':
            self.input_projection = nn.Linear(n_channels, d_model)
            self.model = TransformerEncoder(d_model, depth, heads, mlp_ratio, dropout=dropout)
            self.output_projection = nn.Linear(d_model, n_channels)
        elif self.predictor == 'Series-wise':
            self.input_projection = nn.Linear(seq_len+pred_len, d_model)
            self.model = TransformerEncoder(d_model, depth, heads, mlp_ratio, dropout=dropout)
            self.output_projection = nn.Linear(d_model, pred_len)
        
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, output_attention=False):
        B, L_I, C = x.shape
        
        x = torch.cat([x, self.output_embedding.expand(x.shape[0], -1, -1)], dim=1)
        
        locf = x[:, [-(self.pred_len+1)], :]
        x = x - locf
        
        if self.predictor == 'Series-wise':
            x = x.permute(0, 2, 1)
            
        x = self.input_projection(x)
        x = x + self.positional_embedding[:, 0:x.shape[1], :]
        
        if output_attention:
            x, attn = self.model(x, output_attention=output_attention)
        else:
            x = self.model(x)
        
        if self.predictor == 'Temporal-wise':
            x = x[:, -self.pred_len:, :]
        x = self.output_projection(x)
        
        if self.predictor == 'Series-wise':
            x = x.permute(0, 2, 1)
            
        x = x + locf
        
        if output_attention:
            return x, attn
        
        return x
        