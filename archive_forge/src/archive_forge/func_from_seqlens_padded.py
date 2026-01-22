import math
from dataclasses import dataclass
from typing import (
import torch
@classmethod
def from_seqlens_padded(cls, seqlens: Sequence[int], padding: int) -> '_PaddedSeqLenInfo':
    """
        Input tensors are assumed to be in shape [B, M, *]
        seqstart = padding * torch.arange(batch_size)
        """
    assert not isinstance(seqlens, torch.Tensor)
    assert all((seqlen <= padding for seqlen in seqlens)), f'Seqlens {seqlens} Padding {padding}'
    seqstart_py = list(range(0, len(seqlens) * padding + 1, padding))
    seqlen = torch.tensor(seqlens, dtype=torch.int32)
    return cls(seqlen=seqlen, seqlen_py=seqlens, max_seqlen=max(seqlens), min_seqlen=min(seqlens), seqstart=torch.tensor(seqstart_py, dtype=torch.int32), seqstart_py=seqstart_py, padding=padding)