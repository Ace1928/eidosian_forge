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
@graph_optimization_pass(prerequisites=[comm_fusion_with_concat], apply_after=[])
def schedule_comm_wait(gm: IterGraphModule) -> None:
    """Delay the execution of wait tensors of allreduce until its first user."""
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, 'all_reduce'))
    allreduce_users: Set[fx.Node] = set()
    for allreduce in comm_blocks:
        for output in allreduce.outputs:
            allreduce_users.update(output.users)
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}
    for allreduce in comm_blocks:
        assert len(allreduce.outputs) >= 1, f'Found a allreduce that has zero outputs/users -- {allreduce}.'
        target_node = next(iter(next(iter(allreduce.outputs)).users))
        target_node_index = 2 ** 31
        for user in (user for output in allreduce.outputs for user in output.users):
            index = node_indices[user]
            if index < target_node_index:
                target_node = user
                target_node_index = index
        wait_idx = -1
        for wait_idx, node in enumerate(allreduce.node_list):
            if node == allreduce.wait_nodes[0]:
                break
        assert wait_idx >= 0
        gm.graph.move_before(allreduce.node_list[wait_idx:], target_node)