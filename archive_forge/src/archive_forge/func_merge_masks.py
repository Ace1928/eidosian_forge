import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
    """
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
    mask_type: Optional[int] = None
    merged_mask: Optional[Tensor] = None
    if key_padding_mask is not None:
        mask_type = 1
        merged_mask = key_padding_mask
    if attn_mask is not None:
        batch_size, seq_len, _ = query.shape
        mask_type = 2
        if attn_mask.dim() == 3:
            attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
        else:
            attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
        merged_mask = attn_mask_expanded
        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded + key_padding_mask_expanded
    return (merged_mask, mask_type)