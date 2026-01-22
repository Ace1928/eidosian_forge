from typing import Optional, Tuple, Union
from numbers import Number
import torch
from torch.utils.benchmark import FuzzedTensor
import math
@staticmethod
def sparse_tensor_constructor(size, dtype, sparse_dim, nnz, is_coalesced):
    """sparse_tensor_constructor creates a sparse tensor with coo format.

        Note that when `is_coalesced` is False, the number of elements is doubled but the number of indices
        represents the same amount of number of non zeros `nnz`, i.e, this is virtually the same tensor
        with the same sparsity pattern. Moreover, most of the sparse operation will use coalesce() method
        and what we want here is to get a sparse tensor with the same `nnz` even if this is coalesced or not.

        In the other hand when `is_coalesced` is True the number of elements is reduced in the coalescing process
        by an unclear amount however the probability to generate duplicates indices are low for most of the cases.
        This decision was taken on purpose to maintain the construction cost as low as possible.
        """
    if isinstance(size, Number):
        size = [size] * sparse_dim
    assert all((size[d] > 0 for d in range(sparse_dim))) or nnz == 0, 'invalid arguments'
    v_size = [nnz] + list(size[sparse_dim:])
    if dtype.is_floating_point:
        v = torch.rand(size=v_size, dtype=dtype, device='cpu')
    else:
        v = torch.randint(1, 127, size=v_size, dtype=dtype, device='cpu')
    i = torch.rand(sparse_dim, nnz, device='cpu')
    i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
    i = i.to(torch.long)
    if not is_coalesced:
        v = torch.cat([v, torch.randn_like(v)], 0)
        i = torch.cat([i, i], 1)
    x = torch.sparse_coo_tensor(i, v, torch.Size(size))
    if is_coalesced:
        x = x.coalesce()
    return x