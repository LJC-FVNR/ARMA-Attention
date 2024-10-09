#############################################
# https://github.com/LJC-FVNR/CATS
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.overrides import handle_torch_function, has_torch_function_unary
from torch import _VF
import math
from models.PatchTST import Model as PatchTST
from models.DLinear import Model as DLinear
from models.Autoformer import Model as Autoformer
from models.Informer import Model as Informer
from models.Transformer import Model as Transformer
from models.TimesNet import Model as TimesNet
from models.TiDE import Model as TiDE
from models.FITS import Model as FITS
import copy

def extract_first_one_values(time_series, mask):
    B, L, C = time_series.shape
    indices = torch.arange(L).unsqueeze(0).unsqueeze(-1).expand(B, -1, C).to(time_series.device)
    shifted_mask = torch.cat((torch.zeros(B, 1, C, dtype=mask.dtype).to(time_series.device), mask[:, :-1, :]), dim=1)
    mask_change = (mask == 1) & (shifted_mask == 0)
    first_ones = torch.where(mask_change, indices, L * torch.ones_like(indices))
    first_one_indices = torch.min(first_ones, dim=1).values
    result = torch.gather(time_series, 1, first_one_indices.unsqueeze(1))
    return result

def calculate_weight_decay_loss(modules, weight_decay):
    weight_decay_loss = 0.0
    for module in modules:
        if isinstance(module, nn.Conv1d):
            weight_decay_loss += torch.sum(module.weight ** 2)
        elif hasattr(module, 'children') and len(list(module.children())) > 0:
            for sub_module in module.children():
                if isinstance(sub_module, nn.Conv1d):
                    weight_decay_loss += torch.sum(sub_module.weight ** 2)
    weight_decay_loss *= weight_decay
    return weight_decay_loss

def calculate_ortho_loss(modules, weight_decay):
    weight_decay_loss = 0.0
    for module in modules:
        if isinstance(module, nn.Conv1d):
            w = module.weight
            if len(w.shape) == 2:
                w = w.unsqueeze(0)
            ortho_matrix = torch.matmul(w, w.permute(0,2,1))
            ortho_matrix = ortho_matrix - torch.diag(torch.ones(ortho_matrix.shape[-1], device=ortho_matrix.device)).repeat(ortho_matrix.shape[0], 1, 1)
            weight_decay_loss = weight_decay_loss + ortho_matrix.square().mean()
        elif hasattr(module, 'children') and len(list(module.children())) > 0:
            for sub_module in module.children():
                if isinstance(sub_module, nn.Conv1d):
                    w = sub_module.weight
                    if len(w.shape) == 2:
                        w = w.unsqueeze(0)
                    ortho_matrix = torch.matmul(w, w.permute(0,2,1))
                    ortho_matrix = ortho_matrix - torch.diag(torch.ones(ortho_matrix.shape[-1], device=ortho_matrix.device)).repeat(ortho_matrix.shape[0], 1, 1)
                    weight_decay_loss = weight_decay_loss + ortho_matrix.square().mean()
    weight_decay_loss *= weight_decay
    return weight_decay_loss

def std_excluding_zeros_corrected(tensor, dim):
    """
    Computes the standard deviation along a specified dimension of a tensor, excluding zeros.
    The output tensor has the same shape as the input tensor except the specified dimension is set to 1.

    Args:
    tensor (torch.Tensor): The input tensor.
    dim (int): The dimension along which to compute the standard deviation.

    Returns:
    torch.Tensor: The standard deviation computed along the specified dimension, excluding zeros.
    """
    # Create a mask for non-zero elements
    mask = tensor != 0

    # Calculate sum and count of non-zero elements
    sum_non_zero = torch.sum(tensor * mask, dim=dim, keepdim=True)
    count_non_zero = torch.sum(mask, dim=dim, keepdim=True)

    # Handle case where count is zero to avoid division by zero
    count_non_zero = count_non_zero + (count_non_zero == 0)

    # Calculate mean of non-zero elements
    mean_non_zero = sum_non_zero / count_non_zero

    # Calculate variance of non-zero elements
    variance = torch.sum(((tensor - mean_non_zero) * mask) ** 2, dim=dim, keepdim=True) / count_non_zero

    # Calculate standard deviation
    std_dev = torch.sqrt(variance)

    return std_dev


class Conv1dNonOverlapping(nn.Module):
    def __init__(self, in_channels, out_channels, input_length, kernel_size, groups=1):
        super(Conv1dNonOverlapping, self).__init__()
        extra_length = (kernel_size - (input_length % kernel_size)) % kernel_size
        padding = extra_length // 2
        self.input_length = input_length
        self.groups = groups
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels*kernel_size, kernel_size=kernel_size, stride=kernel_size, padding=padding, groups=groups)
        
    def forward(self, x):
        # x: B, C, L
        x = self.conv1d(x)
        conv_len = x.shape[-1]
        x = x.reshape(x.shape[0], self.out_channels, self.kernel_size, conv_len)  # B G*C Ks Lout
        x = x.permute(0, 1, 3, 2) # B G*C Lout Ks
        x = x.reshape(x.shape[0], self.out_channels, conv_len*self.kernel_size) # B G*C Lout*Ks
        padding = self.input_length - x.shape[-1]
        x = F.pad(x, (padding, 0))
        return x

def weighted_std(tensor, conv_size, weights):
    weights = weights / weights.sum(dim=1, keepdim=True)
    res = []
    for s in conv_size:
        res.append(tensor[:, -s:, :].std(dim=1, keepdim=True))
    return (torch.cat(res, dim=1) * weights).mean(dim=1, keepdim=True)

def ema_3d_weighted(tensor, alpha, partial=None):
    a, b, c = tensor.shape
    b = b if partial is None else partial
    indices = torch.arange(0, b, device=tensor.device)
    indices = indices.view(1, b, 1).repeat(a, 1, c)
    alpha = alpha.view(1, 1, c)
    weights_raw = (1 - alpha) * torch.pow(alpha, indices)
    weights_normalized = b*weights_raw.flip(1) / weights_raw.sum(dim=1, keepdim=True)
    if partial is not None:
        weights_normalized = torch.cat([weights_normalized, weights_normalized[:, [-1], :].repeat(1, tensor.shape[1]-partial, 1)], dim=1)
    return tensor*weights_normalized

def ema_3d(tensor, alpha):
    a, b, c = tensor.shape
    indices = torch.arange(0, b, device=tensor.device)
    indices = indices.view(1, b, 1).repeat(a, 1, c)
    alpha = alpha.view(1, 1, c)
    weights_raw = (1 - alpha) * torch.pow(alpha, indices)
    weights_normalized = weights_raw.flip(1) / weights_raw.sum(dim=1, keepdim=True)
    return (weights_normalized * tensor).sum(dim=1, keepdim=True)

def gate_activation(x, alpha=5):
    return ((torch.nn.functional.gelu(torch.tanh(x)*alpha)/alpha)*(1/0.9640)).abs()

def continuity_loss(tensor):
    B, L, C = tensor.shape

    mean = torch.mean(tensor, dim=1, keepdim=True)
    std = torch.std(tensor, dim=1, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + 1e-6)

    diffs = normalized_tensor[:, 1:, :] - normalized_tensor[:, :-1, :]
    squared_diffs = diffs ** 2
    loss = torch.mean(squared_diffs)
    return loss

Tensor = torch.Tensor

def dropout1d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    Randomly zero out entire channels (a channel is a 1D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out groupedly on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout1d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(dropout1d, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    inp_dim = input.dim()
    if inp_dim not in (2, 3):
        raise RuntimeError(f"dropout1d: Expected 2D or 3D input, but received a {inp_dim}D input. "
                           "Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
                           "spatial dimension, a channel dimension, and an optional batch dimension "
                           "(i.e. 2D or 3D inputs).")

    is_batched = inp_dim == 3
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    result = _VF.feature_dropout_(input, p, training) if inplace else _VF.feature_dropout(input, p, training)

    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)

    return result

class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)

class Dropout1d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 1D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out groupedly on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv1d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout1d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`.
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input).

    Examples::

        >>> m = nn.Dropout1d(p=0.2)
        >>> input = torch.randn(20, 16, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        return dropout1d(input, self.p, self.training, self.inplace)
    
class Dropout1dRand(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        random_rate = torch.rand(1).item()
        return dropout1d(input, random_rate, self.training, self.inplace)
    
class GroupedLinearsUnparallel(nn.Module):
    def __init__(self, in_d, out_d, n=16, zero_init=True, init=0):
        super(GroupedLinears, self).__init__()
        self.model = nn.ModuleList([nn.Linear(in_d, out_d) for i in range(n)])
        [torch.nn.init.zeros_(i.weight) for i in self.model] if zero_init else None	
        self.init = init
        
    def forward(self, x):
        # Input: B L C
        res = []
        for i in range(x.shape[-1]):
            res.append(self.model[i](x[:, :, i]).unsqueeze(2))
        res = torch.cat(res, dim=-1) + self.init
        return res

class GroupedLinears(nn.Module):
    def __init__(self, in_d, out_d, n=16, zero_init=True, init=0, split_ratio=1):
        super(GroupedLinears, self).__init__()
        self.n = n
        self.split_ratio = min(split_ratio, n)
        base_segment_size = n // self.split_ratio
        remainder = n % self.split_ratio
        self.segment_sizes = [base_segment_size + (1 if i < remainder else 0) for i in range(self.split_ratio)]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for segment_size in self.segment_sizes:
            if zero_init:
                self.weights.append(nn.Parameter(torch.zeros(segment_size, in_d, out_d)))
                self.biases.append(nn.Parameter(torch.zeros(segment_size, 1, out_d)))
            else:
                self.weights.append(nn.Parameter(nn.init.trunc_normal_(torch.zeros(segment_size, in_d, out_d), mean=0.0, std=0.01, a=-0.02, b=0.02)))
                self.biases.append(nn.Parameter(torch.zeros(segment_size, 1, out_d)))
        self.init = init

    def forward(self, x):
        # Input: B L C
        outputs = []
        start = 0
        for i in range(self.split_ratio):
            end = start + self.segment_sizes[i]
            x_segment = x[:, :, start:end]  # B L C_segment
            x_segment = x_segment.permute(0, 2, 1).unsqueeze(2)  # B C_segment 1 L
            w = self.weights[i].unsqueeze(0)  # 1 C_segment L Lp
            b = self.biases[i].unsqueeze(0)  # 1 C_segment L Lp
            x_segment = torch.matmul(x_segment, w) + b  # B C_segment 1 Lp
            x_segment = x_segment[:, :, 0, :]  # B C_segment Lp
            x_segment = x_segment.permute(0, 2, 1)  # B Lp C_segment
            outputs.append(x_segment)
            start = end
        x = torch.cat(outputs, dim=2)  # B Lp C
        x = x + self.init
        return x

    
class GroupedLinearsAdvanced(nn.Module):
    def __init__(self, in_d, out_d, grouping, zero_init=True, init=0):
        super(GroupedLinearsAdvanced, self).__init__()
        self.grouping = grouping
        self.groups = len(grouping)
        self.group_weights = nn.ParameterList()
        self.group_biases = nn.ParameterList()

        for group_size in grouping:
            weight = nn.Parameter(0.01*torch.randn(group_size, in_d, out_d))  # Group C L Lp
            bias = nn.Parameter(torch.zeros(group_size, 1, out_d))  # Group C 1 Lp
            torch.nn.init.zeros_(weight) if zero_init else None
            self.group_weights.append(weight)
            self.group_biases.append(bias)

        self.init = init

    def forward(self, x):
        # Input: B L C
        x = x.permute(0, 2, 1)  # B C L
        outputs = []

        start = 0
        for i in range(self.groups):
            end = start + self.grouping[i]
            group_x = x[:, start:end]  # B Group C L
            group_x = group_x.unsqueeze(2)  # B Group 1 C L

            w = self.group_weights[i].unsqueeze(0)  # 1 Group C L Lp
            b = self.group_biases[i].unsqueeze(0)  # 1 Group 1 C Lp

            group_x = torch.matmul(group_x, w) + b  # B Group 1 C Lp
            group_x = group_x.squeeze(2)  # B Group C Lp

            outputs.append(group_x)
            start = end

        x = torch.cat(outputs, dim=1)  # B C Lp
        x = x.permute(0, 2, 1)  # B Lp C
        x = x + self.init
        return x
    
class DefaultPredictor(nn.Module):
    def __init__(self, d_in, d_ff, d_out, n_channels, dropout=0.75):	
        super(DefaultPredictor, self).__init__()
        self.model = nn.Sequential(nn.Linear(d_in, d_ff, bias=True),
                                   nn.Dropout(dropout),
                                   nn.GELU(),
                                   nn.Linear(d_ff, d_out, bias=True))

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, eps=1e-8):	
        # Input: B L C
        x = self.model(x.permute(0,2,1)).permute(0,2,1)
        return x
    
class LinearPredictor(nn.Module):
    def __init__(self, d_in, d_out):	
        super(LinearPredictor, self).__init__()
        self.model = nn.Sequential(nn.Linear(d_in, d_out))
        
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):	
        # Input: B L C
        x = self.model(x.permute(0,2,1)).permute(0,2,1)
        return x
    
class IndependentLinearPredictor(nn.Module):
    def __init__(self, d_in, d_out, n_channels, split_ratio=1):	
        super(IndependentLinearPredictor, self).__init__()
        self.model = GroupedLinears(d_in, d_out, n=n_channels, zero_init=False, init=0, split_ratio=split_ratio)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Input: B L C
        x = self.model(x)
        return x
    
class GroupedLinearPredictor(nn.Module):
    def __init__(self, d_in, d_out, grouping):	
        super(GroupedLinearPredictor, self).__init__()
        self.model = GroupedLinearsAdvanced(d_in, d_out, grouping=grouping, zero_init=False, init=0)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Input: B L C
        x = self.model(x)
        return x
    
class AggregatingPredictor(nn.Module):
    def __init__(self, d_in, d_out, n_channels, kernel_size=16, token_size=8, dropout=0.5, independent=True):
        super(AggregatingPredictor, self).__init__()
        groups = n_channels if independent else 1
        self.agg = torch.nn.Conv1d(n_channels, token_size * n_channels, kernel_size=kernel_size, stride=kernel_size, groups=groups)
        self.token_dropout = Dropout1d(p=dropout, inplace=False)
        self.predictor = torch.nn.Conv1d(d_in//kernel_size, d_out, kernel_size=token_size, stride=token_size)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Input: B L C
        x = x.permute(0,2,1) # B C L
        x = self.agg(x) # B Ts*C L/Ks
        x = self.token_dropout(x)
        x = x.permute(0,2,1) # B L/Ks Ts*C
        x = self.predictor(x) # B Lp C
        return x
    
class MeanAggregationPredictor(nn.Module):
    def __init__(self, pred_len):
        super(MeanAggregationPredictor, self).__init__()
        self.pred_len = pred_len
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Input: B L C
        return x.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
    
class MLPPredictor(nn.Module):
    def __init__(self, d_in, d_ff, d_out, dropout=0.3):	
        super(MLPPredictor, self).__init__()
        self.model = nn.Sequential(nn.Linear(d_in, d_ff),
                                   nn.GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_ff),
                                   nn.GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_out))
        
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):	
        # Input: B L C
        x = self.model(x.permute(0,2,1)).permute(0,2,1)
        return x

class SimplePredictor(nn.Module):
    def __init__(self, d_in, d_out):	
        super(SimplePredictor, self).__init__()
        self.model = nn.Linear(d_in, d_out)
        
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):	
        # Input: B L C
        return self.model(x.permute(0,2,1)).permute(0,2,1)

class ScaledSigmoid(nn.Module):
    def __init__(self, temperature=0.1):
        super(ScaledSigmoid, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        return torch.sigmoid(x / self.temperature)
    
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class SelectiveChannelModule(nn.Module):
    def __init__(self, input_length, input_channels, expanded_channels, drop_channels=1, mlp_ratio=2):
        super(SelectiveChannelModule, self).__init__()
        self.input_channels = input_channels # C
        self.expanded_channels = expanded_channels # N
        self.drop_channels = drop_channels   # C

        self.attention = nn.Sequential(
            nn.Linear(input_channels, int(mlp_ratio*(expanded_channels))),
            #nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(int(mlp_ratio*(expanded_channels)), expanded_channels),
            nn.Sigmoid()
        )
        
        self.aggregation = nn.Sequential(
            nn.Linear(input_length, 1),
            nn.ReLU()
        )
        
        self.sigmoid = nn.Sigmoid()
        self.padding = nn.Parameter(torch.ones(1, 1, drop_channels), requires_grad=False)
    
    def forward(self, x):
        # x: Batch L C
        attention_scores = self.aggregation(x.permute(0,2,1)).permute(0,2,1) # Batch 1 C
        attention_scores = self.attention(attention_scores) # Batch 1 N
        
        # OTS are not affected by the channel sparsity. Their attention scores are set to 1.
        if self.drop_channels > 0:
            attention_scores = torch.cat([attention_scores[:, :, 0:-self.drop_channels], self.padding.repeat(x.shape[0], 1, 1)], dim=-1)
        return attention_scores

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.time_emb_dim = configs.time_emb_dim
        self.in_channels_full = self.in_channels + self.time_emb_dim
        if configs.number_of_targets > 0:
            self.preserving_channels = configs.number_of_targets
        else:
            self.preserving_channels = 1 if configs.features == 'MS' else self.in_channels
        
        self.F_conv_output = configs.F_conv_output
        self.F_noconv_output = configs.F_noconv_output
        self.F_gconv_output_rate = configs.F_gconv_output_rate
        self.F_lin_output = configs.F_lin_output
        self.F_id_output_rate = configs.F_id_output_rate
        self.F_emb_output = configs.F_emb_output
        self.continuity_beta = configs.continuity_beta
        self.channel_sparsity = configs.channel_sparsity
        self.mlp_ratio = configs.mlp_ratio
        self.split_ratio = configs.split_ratio
        
        self.F_gconv_output = int(self.F_gconv_output_rate*self.preserving_channels)
        self.F_id_output = int(self.preserving_channels*self.F_id_output_rate)
        self.padding = 49
        self.kernel_size = self.padding * 2 + 1
        self.predictor = configs.predictor
        self.feature_pool_desc = [self.F_conv_output, 
                                  self.F_conv_output, 
                                  self.F_noconv_output, 
                                  self.F_noconv_output, 
                                  self.F_gconv_output, 
                                  self.F_lin_output, 
                                  self.F_id_output, 
                                  self.F_emb_output,
                                  self.time_emb_dim,
                                  self.in_channels
                                 ]
        self.projection_flag = configs.output_projection
        self.channel_selection = SelectiveChannelModule(input_length=self.seq_len, input_channels=self.in_channels_full, 
                                                        expanded_channels=sum(self.feature_pool_desc)+self.time_emb_dim, 
                                                        drop_channels=0, 
                                                        mlp_ratio=8)
        
        self.pure_emb = nn.Parameter(torch.randn(1, self.seq_len, self.F_emb_output))
        self.feature_independent_flag = 0
        print(f'Current Predictor: {self.predictor}')
        predictor_configs = copy.deepcopy(configs)
        predictor_configs.enc_in = sum(self.feature_pool_desc)
        predictor_configs.dec_in = sum(self.feature_pool_desc)
        predictor_configs.c_out = sum(self.feature_pool_desc)
        if self.predictor == 'Default':
            if self.feature_independent_flag:
                self.Predictor = nn.ModuleList([DefaultPredictor(d_in=self.seq_len, d_ff=4*self.seq_len, d_out=self.pred_len, dropout=0.75) for i in self.feature_pool_desc])
            else:
                self.Predictor = DefaultPredictor(d_in=self.seq_len, d_ff=self.mlp_ratio*self.seq_len, d_out=self.pred_len, n_channels=predictor_configs.enc_in, dropout=configs.predictor_dropout)
        elif self.predictor == 'Simple':
            if self.feature_independent_flag:
                self.Predictor = nn.ModuleList([SimplePredictor(d_in=self.seq_len, d_out=self.pred_len) for i in self.feature_pool_desc])
            else:
                self.Predictor = SimplePredictor(d_in=self.seq_len, d_out=self.pred_len)
        elif self.predictor == 'MLP':
            if self.feature_independent_flag:
                self.Predictor = nn.ModuleList([MLPPredictor(d_in=self.seq_len, d_ff=self.mlp_ratio*(self.seq_len+self.pred_len), d_out=self.pred_len, dropout=0.1) for i in self.feature_pool_desc])
            else:
                self.Predictor = MLPPredictor(d_in=self.seq_len, d_ff=self.mlp_ratio*(self.seq_len+self.pred_len), d_out=self.pred_len, dropout=configs.predictor_dropout)
        elif self.predictor == 'NLinear':
            if self.feature_independent_flag:
                self.Predictor = nn.ModuleList([LinearPredictor(d_in=self.seq_len, d_out=self.pred_len) for i in self.feature_pool_desc])
            else:
                self.Predictor = LinearPredictor(d_in=self.seq_len, d_out=self.pred_len)
        elif self.predictor == 'Agg':
            self.Predictor = AggregatingPredictor(d_in=self.seq_len, d_out=self.pred_len, n_channels=predictor_configs.enc_in, kernel_size=32, token_size=64, dropout=0.5, independent=True)
        elif self.predictor == 'Mean':
            self.Predictor = MeanAggregationPredictor(pred_len=self.pred_len)
        elif self.predictor == 'SimpleIndependent':
            self.Predictor = IndependentLinearPredictor(d_in=self.seq_len, d_out=self.pred_len, n_channels=predictor_configs.enc_in, split_ratio=self.split_ratio)
        elif self.predictor == 'GroupedIndependent':
            self.Predictor = GroupedLinearPredictor(d_in=self.seq_len, d_out=self.pred_len, grouping=self.feature_pool_desc[0:-1] + [1 for i in range(self.in_channels)])
        elif self.predictor == 'PatchTST':
            predictor_configs.e_layers = 3
            predictor_configs.n_heads = 4
            predictor_configs.d_model = 16
            predictor_configs.d_ff = 128
            predictor_configs.dropout = 0.3
            predictor_configs.fc_dropout = 0.3
            predictor_configs.head_dropout = 0
            predictor_configs.patch_len = 16
            predictor_configs.stride = 8
            predictor_configs.revin = 1
            self.Predictor = PatchTST(predictor_configs)
        elif self.predictor == 'DLinear':
            self.Predictor = DLinear(predictor_configs)
        elif self.predictor == 'Autoformer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Autoformer(predictor_configs)
        elif self.predictor == 'Informer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Informer(predictor_configs)
        elif self.predictor == 'TimesNet':
            predictor_configs.label_len = 48
            predictor_configs.factor = 3
            self.Predictor = TimesNet(predictor_configs)
        elif self.predictor == 'Transformer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Transformer(predictor_configs)
        elif self.predictor == 'TiDE':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 2
            predictor_configs.label_len = 48
            predictor_configs.factor = 3
            predictor_configs.d_model = 256
            predictor_configs.d_ff = 256
            predictor_configs.dropout = 0.3
            self.Predictor = TiDE(predictor_configs)
        elif self.predictor == 'FITS':
            predictor_configs.individual = 0
            self.Predictor = FITS(predictor_configs)
            
        self.predictor_configs = predictor_configs
        self.input_conv = Conv1dNonOverlapping(in_channels=self.in_channels_full, out_channels=self.F_noconv_output, input_length=self.seq_len, kernel_size=12, groups=1)
        self.input_conv_alter = Conv1dNonOverlapping(in_channels=self.in_channels_full, out_channels=self.F_noconv_output, input_length=self.seq_len, kernel_size=24, groups=1)
        self.input_conv2 = nn.Conv1d(in_channels=self.in_channels_full, out_channels=self.F_conv_output, kernel_size=49, padding=24)
        self.input_conv2_alter = nn.Conv1d(in_channels=self.in_channels_full, out_channels=self.F_conv_output, kernel_size=193, padding=96)
        self.input_conv_grouped2 = nn.Conv1d(in_channels=self.preserving_channels, out_channels=self.F_gconv_output, kernel_size=self.kernel_size, padding=self.padding, groups=self.preserving_channels)
        self.input_conv_1x1 = nn.Conv1d(in_channels=self.in_channels_full, out_channels=self.F_lin_output, kernel_size=1, padding=0)
        self.output_conv_1x1 = nn.Sequential(nn.Identity(),
                                             nn.Conv1d(in_channels=predictor_configs.dec_in+self.time_emb_dim, out_channels=self.preserving_channels, kernel_size=1, padding=0))
        torch.nn.init.zeros_(self.output_conv_1x1[-1].weight)
        self.random_drop = nn.Identity()
        
        self.init_gated = 0
        if self.init_gated: 
            size = 2
            self.init_generate_self_mask = GroupedLinears(self.seq_len, 1, n=self.in_channels, zero_init=True, init=(1/size))
            self.init_mask_generate_linspace = nn.Parameter(torch.linspace(1, 0, steps=self.seq_len), requires_grad=False)
            self.init_mask_generate_gelu = nn.LeakyReLU(0.1)
        
        self.gated = configs.mid_gate
        if self.gated: 
            size = 2
            self.generate_self_mask = GroupedLinears(self.seq_len, 1, n=predictor_configs.enc_in, zero_init=True, init=0, split_ratio=1)
            self.mask_generate_linspace = nn.Parameter(torch.linspace(1, 0, steps=self.seq_len), requires_grad=False)
            self.mask_generate_gelu = nn.Identity()
        
        self.regroup = False
        self.preprocessing_method = 'lastvalue'
        self.preprocessing_detach = True
        self.preprocessing_affine = False
        self.channel_w = nn.Parameter(torch.ones(self.preserving_channels), requires_grad=True)
        self.channel_b = nn.Parameter(torch.zeros(self.preserving_channels), requires_grad=True)
        self.eps = 1e-7

        if self.preprocessing_method == 'auel':
            self.conv_size = [8, 16, 64]#[49, 145, 385]
            self.ema_alpha = nn.Parameter(0.9*torch.ones(self.in_channels))
            self.std_weights = nn.Parameter(torch.cat([0.5*torch.ones(len(self.conv_size), self.in_channels), 
                                                          torch.ones(1, self.in_channels)]))

    
    def gate(self, x, activation_func, generated_mask, linspace, return_mask=False):
        mask_infer = -activation_func(generated_mask(x)) # B 1 C, a, a \in (-inf, 0)
        gate_len = x.shape[1]
        mask_infer = linspace.reshape(1, gate_len, 1).repeat(mask_infer.shape[0], 1, mask_infer.shape[-1]) * mask_infer + 1 # ax+1
        mask_stop = mask_infer - mask_infer.detach() + (mask_infer > 0).int().detach()
        x = x * mask_stop
        if return_mask:
            return x, mask_stop
        else:
            return x

    def encoder(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        
        x = torch.cat([
            self.input_conv(x.permute(0,2,1)).permute(0,2,1) if self.F_noconv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv_alter(x.permute(0,2,1)).permute(0,2,1) if self.F_noconv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv2(x.permute(0,2,1)).permute(0,2,1) if self.F_conv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv2_alter(x.permute(0,2,1)).permute(0,2,1) if self.F_conv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv_grouped2(x[:, :, -self.preserving_channels:].permute(0,2,1)).permute(0,2,1) if self.F_gconv_output_rate > 0 else torch.Tensor([]).to(x.device),
            self.input_conv_1x1(x.permute(0,2,1)).permute(0,2,1) if self.F_lin_output > 0 else torch.Tensor([]).to(x.device),
            *[x[:, :, -self.preserving_channels:] for i in range(self.F_id_output_rate)],
            self.pure_emb.repeat(x.shape[0], 1, 1),
            x
        ], dim=-1) # B LI latent+C
        return x
    
    def predictor_(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.feature_independent_flag:
            csp = 0 # current start point
            x_pred = []
            for index, value in enumerate(self.feature_pool_desc):
                x_pred.append(self.Predictor[index](x[:, :, csp:csp+value], x_mark_enc[:, :, csp:csp+value], x_dec[:, :, csp:csp+value], x_mark_dec[:, :, csp:csp+value]))
                csp += value
            x_pred = torch.cat(x_pred, dim=-1)
        else:
            x_pred = self.Predictor(x.clone(), x_mark_enc, x_dec, x_mark_dec)
        x = torch.cat([x.clone(), x_pred], dim=1) # B L latent
        return x
    
    def regroup(self, x):
        # regroup
        x_reg = []
        x_conv = x[:, :, 0:self.latent_channels_rate*self.in_channels]
        x_ind = x[:, :, self.latent_channels_rate*self.in_channels:self.latent_channels_rate*self.in_channels+self.latent_channels_grouped_rate*self.in_channels]
        x_orig = x[:, :, -self.in_channels:]
        for i in range(self.in_channels):
            x_reg.append(x_conv[:, :, i*self.latent_channels_rate:(i+1)*self.latent_channels_rate])
            x_reg.append(x_ind[:, :, i*self.latent_channels_grouped_rate:(i+1)*self.latent_channels_grouped_rate])
            x_reg.append(x_orig[:, :, [i]])
        x = torch.cat(x_reg, dim=-1)
        return x

    def decoder(self, x, channel_attn_score_decode):
        if self.projection_flag:
            x_resid = self.output_conv_1x1((x*channel_attn_score_decode).permute(0,2,1)).permute(0,2,1) # B L C
            x_clone_1 = x.clone()
            x_clone_1[:, -self.pred_len:, -self.preserving_channels:] = x[:, -self.pred_len:, -self.preserving_channels:] + x_resid[:, -self.pred_len:, :]
            output = x_clone_1
        else:
            output = x
        return output
    
    def preprocessing(self, x, x_output=None, cutoff=None, preprocessing_method=None, preprocessing_detach=None, preprocessing_affine=None):
        eps = self.eps
        x_orig = x.clone()
        if preprocessing_method is None: preprocessing_method = self.preprocessing_method
        if preprocessing_detach is None: preprocessing_detach = self.preprocessing_detach
        if preprocessing_affine is None: preprocessing_affine = self.preprocessing_affine
        if self.preprocessing_affine:
            x = x * self.channel_w + self.channel_b
        if preprocessing_method == 'lastvalue':
            mean = x[:,-1:,:]
            std = torch.ones(1, 1, x.shape[-1]).to(x.device)
        elif preprocessing_method == 'standardization':
            if cutoff is None:
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True) + eps
            else:
                n_elements = cutoff.sum(dim=1, keepdim=True) # B L C -> B 1 C
                # x = x * cutoff # already done
                mean = x.sum(dim=1, keepdim=True) / (n_elements+eps).detach()
                sst = (cutoff*((torch.square(x - mean))**2)).sum(dim=1, keepdim=True)
                std = torch.sqrt(sst / (n_elements-1+eps)) + eps
                std = std.detach()
                std = torch.ones(1, 1, x.shape[-1]).to(x.device)
        elif preprocessing_method == 'auel':
            anchor_context = ema_3d(x[:, 0:(self.seq_len), :], alpha=self.ema_alpha[0:x.shape[-1]].clip(0.000001, 0.999999))
            std = weighted_std(x[:, 0:(self.seq_len), :], self.conv_size+[self.seq_len], gate_activation(self.std_weights*2).unsqueeze(0).repeat(x.shape[0], 1, 1)[:, :, 0:x.shape[-1]]).clip(1e-4, 5000)+eps
            mean = anchor_context
        elif preprocessing_method == 'none':
            mean = torch.zeros(1, 1, x.shape[-1]).to(x.device)
            std = torch.ones(1, 1, x.shape[-1]).to(x.device)
        if x_output is None:
            x_output = x_orig.clone()
        if self.preprocessing_detach:
            x_output = (x_output - mean.detach())/std.detach()
        else:
            x_output = (x_output - mean)/std
        if self.preprocessing_affine:
            x_output = (x_output - self.channel_b) / self.channel_w
        return x_output, mean, std

    def inverse_processing(self, x, mean, std):
        x = x.clone()
        if self.preprocessing_affine:
            x = (x - self.channel_b)/self.channel_w
        x = x*std + mean
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, return_mask=False, train=False):
        # x_enc: [Batch, Input length, Channel]
        # Preprocessing
        output_masks = []
        x_for_init = x_enc.clone()
        if self.init_gated:
            x_for_init, init_mask = self.gate(x_for_init, self.init_mask_generate_gelu, self.init_generate_self_mask, self.init_mask_generate_linspace, return_mask=True)
            if return_mask:
                output_masks.append(init_mask)
        else:
            init_mask = None
        x_preprocessed, mean, std = self.preprocessing(x_for_init[:, :, -self.preserving_channels:], x_output=x_enc[:, :, -self.preserving_channels:], cutoff=init_mask)
        x = torch.cat([x_for_init[:, :, 0:-self.preserving_channels], x_preprocessed], dim=-1)
 
        # Add Temporal Embedding and Default Embedding for Encoder
        if self.time_emb_dim:
            x_enc = torch.cat([x_mark_enc, x], dim=-1)
        else:
            x_enc = x
        
        # Encoder
        channel_attn_score = self.channel_selection(x_enc) if self.channel_sparsity else torch.ones(1, 1, sum(self.feature_pool_desc)+self.time_emb_dim, device=x_enc.device)
        channel_attn_score_decode = torch.cat([torch.ones_like(channel_attn_score[:, :, 0:-(self.preserving_channels+self.time_emb_dim)]).to(x.device), channel_attn_score[:, :, -(self.preserving_channels+self.time_emb_dim):]], dim=-1).clone()
        channel_attn_score = torch.cat([channel_attn_score[:, :, 0:-(self.preserving_channels+self.time_emb_dim)], torch.ones(channel_attn_score.shape[0], 1, self.preserving_channels, device=x_enc.device)], dim=-1)
        x = self.encoder(x_enc)
        
        x = channel_attn_score * x
        add_loss_cnt = continuity_loss(x)*self.continuity_beta
        cnt_loss = add_loss_cnt
        
        if self.gated:
            if return_mask:
                x, mid_mask = self.gate(x, self.mask_generate_gelu, self.generate_self_mask, self.mask_generate_linspace, return_mask=True)
            else:
                x = self.gate(x, self.mask_generate_gelu, self.generate_self_mask, self.mask_generate_linspace)
        
        # Align Input Format for Different Predictors
        x_dec = torch.zeros(x.shape[0], self.pred_len, x.shape[-1], device=x.device)
        if self.predictor_configs.label_len:
            x_dec = torch.cat([x[:, -self.predictor_configs.label_len:, :], x_dec], dim=1)
        
        # Predictor
        x = self.predictor_(x, x_mark_enc, x_dec, x_mark_dec)
        
        x_predictor_snapshot = x.detach().clone()
        x_predictor_snapshot[:, :, -self.preserving_channels:] = self.inverse_processing(x_predictor_snapshot[:, :, -self.preserving_channels:], mean, std).detach()
        
        # Add Temporal Embedding Again
        if self.time_emb_dim:
            timestamp_embedding = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
            x = torch.cat([timestamp_embedding, x], dim=-1)
        
        # Random Channel Dropout and Default Embedding for Decoder
        x = self.random_drop(x.permute(0,2,1)).permute(0,2,1) #+ self.dec_emb
        if self.regroup:
            x = self.regroup(x)
        
        x_dec = self.decoder(x, channel_attn_score_decode)
        
        # Inverse Processing
        x = x_dec[:, -self.pred_len:, -self.in_channels:]
        x[:, :, -self.preserving_channels:] = self.inverse_processing(x[:, :, -self.preserving_channels:], mean, std)
        
        output_masks = {}
        if self.gated and return_mask:
            output_masks['mid'] = mid_mask
            
        if train:
            return x, cnt_loss
        
        if return_mask:
            return x, output_masks, x_predictor_snapshot
        else:
            return x # [Batch, Output length, Channel]