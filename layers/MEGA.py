##########################################################################
# FROM https://github.com/facebookresearch/mega/blob/main/fairseq/modules/moving_average_gated_attention.py#L24
##########################################################################

import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

class FairseqDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, batch_first: bool = False, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    'Cannot enable dropout during inference for module {} '
                    'because module_name was not set'.format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    'Enabling dropout during inference for module: {}'.format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)


class FairseqFeatureDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, batch_first: bool = False, inplace: bool = False):
        if self.training or self.apply_during_inference:
            if batch_first:
                # B x L x D -> B x D x L -> B x L x D
                return F.dropout2d(x.transpose(-1, -2), p=self.p, training=True, inplace=inplace).transpose(-1, -2)
            else:
                assert x.dim() == 3
                # L x B x D -> B x D x L -> L x B x D
                return F.dropout2d(x.permute(1, 2, 0), p=self.p, training=True, inplace=inplace).permute(2, 0, 1)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    'Cannot enable dropout during inference for module {} '
                    'because module_name was not set'.format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    'Enabling dropout during inference for module: {}'.format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)
    
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

class SequenceNorm(nn.Module):
    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        super().__init__()
        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(embedding_dim) #LayerNorm(embedding_dim, eps=eps, elementwise_affine=affine, export=export)
        elif norm_type == 'scalenorm':
            self.norm = ScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embedding_dim, eps=eps)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'syncbatchnorm':
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def normalize(self, x):
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            assert x.dim() == 3
            x = x.permute(1, 2, 0)
            x = self.norm(x)
            return x.permute(2, 0, 1)
        else:
            return self.norm(x)

    def forward(self, x):
        return self.normalize(x)
    
class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.

    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = torch.sigmoid(self.delta)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        kernel = (p * self.beta) * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q = self.coeffs()
        # D x N x L+1
        vander = torch.arange(length + 1).to(p).view(1, 1, length + 1) * torch.log(q)
        vander = torch.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * (self.gamma * self.scale).unsqueeze(-1)
            ox = torch.einsum('bdn,dnl->bdl', hx, k)
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = (p * self.beta) * vander
        k = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

        k_f = torch.fft.rfft(k.float(), n=2 * length)
        x_f = torch.fft.rfft(x.float(), n=2 * length)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * length)[..., 0:length]
        out = out.type_as(x)
        if ox is not None:
            out = out + ox

        h = torch.einsum('bdl,dnl->bdn', x, torch.flip(kernel, dims=[2]))
        if hh is not None:
            h = h + hh
        # L x B x D, B x D x N
        return out.permute(2, 0, 1), h

    def one_step(self, x, hx=None):
        p, q = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = (p * self.beta).squeeze(-1) * x
        if hx is not None:
            h = h + q.squeeze(-1) * hx
        # B x D
        out = torch.einsum('bdn,dn->bd', h, self.gamma * self.scale)
        # 1 x B x D, B x D x N
        return out.unsqueeze(0), h

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional EMA does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
            # B x D -> 1 x B x D
            out = F.silu(out + residual)
        else:
            # D x L
            k = self.kernel(seq_len)
            fft_len = seq_len
            s = 0
            kernel_size = k.size(1)
            if self.bidirectional:
                k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
                # D x 2*L-1
                k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                x = F.pad(x, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s = 2 * kernel_size - 2

            k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
            x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
            # B x D x L
            out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
            out = out.type_as(x)
            # B x D x L -> L x B x D
            out = F.silu(out.permute(2, 0, 1) + residual)

        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "ema_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "ema_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}'.format(self.embed_dim, self.ndim, self.bidirectional, self.truncation)
    
class MovingAverageGatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        hdim,
        ndim,
        dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        activation='silu',
        attention_activation='softmax',
        bidirectional=False,
        chunk_size=-1,
        truncation=None,
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        rel_pos_bias='simple',
        max_positions=1024,
        export=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim # 512
        self.hdim = hdim # 1024
        self.zdim = zdim # 128
        self.ndim = ndim # 16
        self.activation = nn.SiLU() # utils.get_activation_fn(activation=activation)
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)
        # Attention dropout is standard dropout
        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)

        self.chunk_size = chunk_size
        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)

        self.move = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)

        self.v_proj = nn.Linear(embed_dim, hdim)
        self.mx_proj = nn.Linear(embed_dim, zdim + hdim + 2 * embed_dim)
        self.h_proj = nn.Linear(hdim, embed_dim)

        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        max_positions = max_positions if chunk_size < 0 else chunk_size

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    def element_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        if padding_mask is not None:
            # B x K x C
            inverse_mask = 1.0 - padding_mask.type_as(q)
            # B x K x 1
            lengths = inverse_mask.sum(dim=-1, keepdim=True)
            # B x K x 1 x 1
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            lengths = slen
            inverse_mask = None

        if attn_mask is not None:
            # C x 1
            lengths = attn_mask.sum(dim=-1, keepdim=True)

        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) / lengths

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = utils.relu2(qk).type_as(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = utils.laplace(qk).type_as(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if inverse_mask is not None:
            attn_weights = attn_weights * inverse_mask.unsqueeze(2)

        if attn_mask is not None:
            attn_weights = attn_weights * attn_mask

        return attn_weights

    def softmax_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        # scaled attention
        q = q * self.scaling
        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3))

        if attn_mask is not None:
            qk = qk + attn_mask
        else:
            _, _, L, S = qk.shape
            attn_bias = torch.zeros(L, S, dtype=q.dtype)
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float('-inf'))
            attn_bias = attn_bias.to(q.dtype).to(q.device)
            qk = qk + attn_bias

        if padding_mask is not None:
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = F.softmax(qk, dim=-1).type_as(qk)
        return attn_weights

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_attn_fn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        residual = x
        # if self.prenorm:
        #     x = self.norm(x)

        # L x B x E
        v = self.activation(self.v_proj(x))

        # L x B x D
        mx = self.move(x, padding_mask, incremental_state)
        mx = self.dropout(mx)

        # L x B x D -> L x B x (2*D+S+E)
        base = self.mx_proj(mx)
        u, zr, hx = torch.split(base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], dim=-1)
        # L x B x D
        u = torch.sigmoid(u)
        # L x B x (E+S)
        z, r = torch.split(F.silu(zr), [self.zdim, self.hdim], dim=-1)
        # L x B x S -> L x B x 1 x S -> L x B x 2 x S
        z = z.unsqueeze(2) * self.gamma + self.beta
        # L x B x 2 x S -> L x B x S
        q, k = torch.unbind(z, dim=2)

        # L x B x D -> B x L x D
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        if saved_state is not None:
            # assert self.chunk_size < 0 or q.size(1) <= self.chunk_size
            # saved states are stored with shape (bsz, seq_len, dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
            prev_padding_mask: Optional[Tensor] = None
            if "prev_padding_mask" in saved_state:
                prev_padding_mask = saved_state["prev_padding_mask"]
            padding_mask = MovingAverageGatedAttention._append_prev_padding_mask(
                padding_mask=padding_mask,
                prev_padding_mask=prev_padding_mask,
                batch_size=bsz,
                seq_len=k.size(1),
            )

            if self.chunk_size < 0:
                saved_state["prev_key"] = k
                saved_state["prev_value"] = v
                saved_state["prev_key_padding_mask"] = padding_mask
            else:
                curr_len = k.size(1) % self.chunk_size
                if curr_len == 0:
                    if "prev_key" in saved_state:
                        del saved_state["prev_key"]
                        del saved_state["prev_value"]
                        del saved_state["prev_key_padding_mask"]
                else:
                    saved_state["prev_key"] = k
                    saved_state["prev_value"] = v
                    saved_state["prev_key_padding_mask"] = padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

        ctx_len = k.size(1)
        if self.chunk_size < 0:
            # B x L x S -> B x 1 x L x S
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            if padding_mask is not None:
                # B x L -> B x 1 x L
                padding_mask = padding_mask.unsqueeze(1)
        else:
            if seq_len < self.chunk_size:
                q = q.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = seq_len // self.chunk_size
                q = q.reshape(bsz, nc, self.chunk_size, self.zdim)

            if ctx_len < self.chunk_size:
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = ctx_len // self.chunk_size
                k = k.reshape(bsz, nc, self.chunk_size, self.zdim)
                v = v.reshape(bsz, nc, self.chunk_size, self.hdim)
                if padding_mask is not None:
                    # B x L -> B x K x C
                    padding_mask = padding_mask.view(bsz, nc, self.chunk_size)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, padding_mask, attn_mask, before_attn_fn)

        if before_attn_fn:
            return attn_weights, v

        v = self.hidden_dropout(v, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        # B x K x C x E -> B x L x E -> L x B x E
        h = torch.matmul(kernel, v).view(bsz, seq_len, self.hdim).transpose(0, 1)
        # L x B x E -> L x B x D
        h = self.activation(hx + self.h_proj(h * r))
        h = self.dropout(h)
        # L x B x D
        out = torch.addcmul(residual, u, h - residual)

        # if not self.prenorm:
        #     out = self.norm(out)

        return out