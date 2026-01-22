import math
from dataclasses import dataclass
from typing import (
import torch
@classmethod
def from_tensor_lists_qkv(cls, tensors_q: Sequence[torch.Tensor], tensors_k: Sequence[torch.Tensor], tensors_v: Optional[Sequence[torch.Tensor]]=None) -> Tuple['BlockDiagonalMask', torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert len(tensors_q) == len(tensors_k)
    assert tensors_v is None or len(tensors_v) == len(tensors_q)
    batch_sizes = [tensor.shape[0] for tensor in tensors_q]
    q_seqlens, kv_seqlens = ([], [])
    for i, (q, k) in enumerate(zip(tensors_q, tensors_k)):
        assert q.shape[0] == k.shape[0]
        q_seqlens += [q.shape[1]] * q.shape[0]
        kv_seqlens += [k.shape[1]] * k.shape[0]
        assert tensors_v is None or tensors_v[i].shape[:2] == k.shape[:2]
    block_diag = cls.from_seqlens(q_seqlens, kv_seqlens)
    block_diag._batch_sizes = batch_sizes
    return (block_diag, torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_q], dim=1), torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_k], dim=1), torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_v], dim=1) if tensors_v is not None else None)