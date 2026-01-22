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
@graph_optimization_pass(prerequisites=[remove_copy_from_optimizer], apply_after=[schedule_comm_wait])
def iter_move_grads_and_optimizers(gm: IterGraphModule, target_comm_node: str, target_dest_node: str) -> None:
    """Extract a comm block and split out a new optimizer and step for it.

    This subgraph is then moved to the forward graph.
    """
    for comm_block in get_all_comm_blocks(gm, 'all_reduce'):
        if comm_block.comm_node.name == target_comm_node:
            break
    else:
        raise ValueError(f'Cannot find {target_comm_node}')
    optim_blocks = get_all_fused_optimizer_blocks(gm, '_fused_adam')
    for optim_block in optim_blocks:
        optim_args = AdamArgs(*optim_block.optim.optim_node.args)
        one_output = next(iter(comm_block.outputs))
        if one_output in optim_args.grads:
            break
    else:
        raise ValueError(f'{target_comm_node} is not used by any fused optimizer.')
    move_optim, _ = split_fused_optimizer(gm, optim_block, comm_block.outputs)
    move_nodes = find_all_descendants(gm, [comm_block.comm_node, move_optim.step.add_node])
    stop_node = find_node(gm.graph, lambda n: n.name == target_dest_node)[0]
    gm.graph.move_to_next_iter_before(move_nodes, stop_node)