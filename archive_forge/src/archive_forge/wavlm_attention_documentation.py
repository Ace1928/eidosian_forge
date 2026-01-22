import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor

        Args:
            query (Tensor): Input of shape ``(batch_size, src_len, embed_dim)``.
            key_padding_mask (Tensor or None, optional): Mask to exclude keys that are pads, of shape
                `(batch, src_len)`, where padding elements are indicated by 1s. (Default: ``None``)
            attn_mask: Needs to be ``None``. The argument exists for compatibility with
                ``EncoderLayer``. (Default: ``None``)
            position_bias (Tensor or None, optional): Position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``. When used inside WavLM model encoder, will be
                generated in the first layer and then passed from each encoder layer to the next one.
                (Default: ``None``)
        Returns:
            attn_output (Tensor): Attention output of shape ``(batch_size, src_len, embed_dim)``.
            position_bias (Tensor or None): Position bias of shape ``(batch_size * num_heads, src_len, src_len)``.
        