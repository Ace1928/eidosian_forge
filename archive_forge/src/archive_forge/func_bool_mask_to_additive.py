from typing import Optional
import torch
def bool_mask_to_additive(mask: torch.Tensor, dtype: Optional[torch.dtype]=torch.float32) -> torch.Tensor:
    assert mask.dtype == torch.bool, 'This util is meant to convert in between bool masks and additive ones'
    mask_ = torch.zeros_like(mask, dtype=dtype)
    mask_[~mask] = float('-inf')
    return mask_