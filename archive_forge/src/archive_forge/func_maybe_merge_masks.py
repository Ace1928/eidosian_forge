from typing import Optional
import torch
def maybe_merge_masks(att_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], batch_size: int, src_len: int, num_heads: int, tgt_len: Optional[int]=None) -> Optional[torch.Tensor]:
    if tgt_len is None:
        tgt_len = src_len
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (batch_size, src_len)
        key_padding_mask = _reshape_key_padding_mask(key_padding_mask, batch_size, src_len, num_heads)
        if att_mask is None:
            att_mask = key_padding_mask.expand(-1, tgt_len, -1)
        elif att_mask.dtype == torch.bool:
            att_mask = att_mask.logical_and(key_padding_mask)
        else:
            att_mask = att_mask.masked_fill(~key_padding_mask, float('-inf'))
    return att_mask