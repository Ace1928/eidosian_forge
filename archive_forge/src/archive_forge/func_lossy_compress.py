from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def lossy_compress(self, dense: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """From dense tensor to lossy reconstruction of dense tensor with the help of SST and DST
        tensor calculation. If requested sparsity is zero (or top_100_percent) then simply returns
        the input dense tensor as the reconstruction.

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).

        Returns:
            (Tuple[Tensor, Tensor, Tensor]):
                A tuple of the form (lossy_reconstruction, sst, dst) with three tensors of the same
                shape as the dense tensor.
        """
    if _is_sparsity_zero(dense, self._sst_top_k_percent, self._sst_top_k_element, self._sst_top_k_dim) and _is_sparsity_zero(dense, self._dst_top_k_percent, self._dst_top_k_element, self._dst_top_k_dim):
        return (dense, None, dense)
    else:
        sst = self.dense_to_sst(dense)
        dst = self.dense_sst_to_dst(dense, sst)
        return (self.sst_dst_to_dense(sst, dst), sst, dst)