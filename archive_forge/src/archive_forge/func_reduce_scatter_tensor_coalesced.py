import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, Optional, cast, TYPE_CHECKING
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
from torch.fx.experimental.proxy_tensor import (
from torch._custom_ops import impl_abstract
from torch.distributed.distributed_c10d import (
def reduce_scatter_tensor_coalesced(inputs: List[torch.Tensor], reduceOp: str, scatter_dim: List[int], group: RANK_TYPES, tag: str='') -> List[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    The input tensors are left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    tag, rankset, group_size = _expand_group(group, tag)
    assert len(scatter_dim) == len(inputs)
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        assert tensor.size(dim) % group_size == 0, f'input dimension {dim} ({tensor.size(dim)} must be a multiple of group_size {group_size} for tensor at index {idx}'
        if dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=dim)
            inputs[idx] = torch.cat(tensor_list)
    tensor_list = torch.ops.c10d_functional.reduce_scatter_tensor_coalesced(inputs, reduceOp, tag, rankset, group_size)
    return list(map(_maybe_wrap_tensor, tensor_list))