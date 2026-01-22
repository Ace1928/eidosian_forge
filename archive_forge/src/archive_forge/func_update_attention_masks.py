import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
@torch.inference_mode()
def update_attention_masks(attention_masks: torch.Tensor, ancestors: torch.Tensor) -> torch.Tensor:
    """Expand the attention masks.

    Parameters
    ----------
    attention_masks
        The attention masks for each sequence in the batch.
    ancestors
        The sequences to which the token ids need to be added.

    Returns
    -------
    The attention masks padded with 1s.

    """
    attention_masks = torch.index_select(attention_masks, 0, ancestors)
    return torch.concatenate([attention_masks, torch.ones(attention_masks.shape[:-1] + (1,), device=attention_masks.device)], axis=-1)