import math
from dataclasses import dataclass
from typing import (
import torch
@classmethod
def from_seqlens(cls, q_seqlen: Sequence[int], kv_seqstarts: Sequence[int], kv_seqlen: Sequence[int]) -> 'BlockDiagonalGappyKeysMask':
    """Creates a :attr:`BlockDiagonalPaddedKeysMask` from a list of tensor
        lengths for query and key/value.

        Args:
            q_seqlen (Sequence[int]): List or tensor of sequence lengths for query tensors
            kv_padding (int): Padding for k/v - also an upperbound on each individual key length
            kv_seqlen (Sequence[int]): List or tensor of sequence lengths for key/value.
            causal_diagonal: unused, for BC only
        Returns:
            BlockDiagonalGappyKeysMask
        """
    assert len(q_seqlen) == len(kv_seqlen), (q_seqlen, kv_seqlen)
    q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
    k_seqinfo = _GappySeqInfo.from_seqlens_gappy(kv_seqstarts, kv_seqlen)
    return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)