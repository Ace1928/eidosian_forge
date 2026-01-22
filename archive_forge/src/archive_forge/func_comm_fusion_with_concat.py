import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
@graph_optimization_pass(prerequisites=[], apply_after=[])
def comm_fusion_with_concat(gm: IterGraphModule, bucket_size_mb: int) -> None:
    """Run fuse communication with concat.

    This implementation uses concat to concat the bucketed gradients.
    """
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, 'all_reduce'))
    _expedite_comm_ops(gm, comm_blocks)
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, 'all_reduce'))
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}
    bucket_size = 1 * 1024 ** 2
    bucket_cap_size = bucket_size_mb * 1024 ** 2
    begin = end = curr_size = 0
    while end < len(comm_blocks):
        curr_size += cast(torch.Size, comm_blocks[end].shape).numel() * 4
        end += 1
        if curr_size < bucket_size:
            continue
        _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)
        bucket_size = bucket_cap_size
        begin = end
        curr_size = 0
    else:
        if begin < len(comm_blocks):
            _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)