from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def sst_dst_to_dense(self, sst: Optional[Tensor], dst: Optional[Tensor]=None) -> Tensor:
    """From SST and DST returns a dense reconstructed tensor (RT). When argument dst=None, simply returns
        the inverse transform of the SST tensor.

        Args:
            sst (Tensor):
                Singal sparse tensor. Required argument.
            dst (Tensor, optional):
                Delta sparse tensor, optional.

        Returns:
            (Tensor):
                A dense tensor in real number domain from the SST.
        """
    assert not (sst is None and dst is None), 'both-None-case is not useful'
    if sst is None:
        return dst
    dense_rt = torch.real(self._inverse_transform(sst, dim=self._sst_top_k_dim))
    if dst is not None:
        dense_rt += dst
    return dense_rt