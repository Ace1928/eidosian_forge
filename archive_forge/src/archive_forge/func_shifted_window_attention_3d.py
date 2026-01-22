from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
def shifted_window_attention_3d(input: Tensor, qkv_weight: Tensor, proj_weight: Tensor, relative_position_bias: Tensor, window_size: List[int], num_heads: int, shift_size: List[int], attention_dropout: float=0.0, dropout: float=0.0, qkv_bias: Optional[Tensor]=None, proj_bias: Optional[Tensor]=None, training: bool=True) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, T, H, W, C]): The input tensor, 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): 3-dimensions window size, T, H, W .
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention (T, H, W).
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[B, T, H, W, C]: The output tensor after shifted window attention.
    """
    b, t, h, w, c = input.shape
    pad_size = _compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    num_windows = padded_size[0] // window_size[0] * (padded_size[1] // window_size[1]) * (padded_size[2] // window_size[2])
    x = x.view(b, padded_size[0] // window_size[0], window_size[0], padded_size[1] // window_size[1], window_size[1], padded_size[2] // window_size[2], window_size[2], c)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(b * num_windows, window_size[0] * window_size[1] * window_size[2], c)
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q = q * (c // num_heads) ** (-0.5)
    attn = q.matmul(k.transpose(-2, -1))
    attn = attn + relative_position_bias
    if sum(shift_size) > 0:
        attn_mask = _compute_attention_mask_3d(x, (padded_size[0], padded_size[1], padded_size[2]), (window_size[0], window_size[1], window_size[2]), (shift_size[0], shift_size[1], shift_size[2]))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)
    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)
    x = x.view(b, padded_size[0] // window_size[0], padded_size[1] // window_size[1], padded_size[2] // window_size[2], window_size[0], window_size[1], window_size[2], c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
    x = x[:, :t, :h, :w, :].contiguous()
    return x