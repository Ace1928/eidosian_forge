import torch
from torch.distributed._shard.metadata import ShardMetadata
from typing import Sequence
def narrow_tensor_by_index(tensor: torch.Tensor, offsets: Sequence[int], sizes: Sequence[int]) -> torch.Tensor:
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor