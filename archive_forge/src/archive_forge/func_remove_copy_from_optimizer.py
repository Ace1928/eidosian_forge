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
def remove_copy_from_optimizer(gm: IterGraphModule) -> None:
    """Erase the orphant copy_ that generated when tracing optimizer.

    Two reasons why we could not simply use the DCE of fx.Graph.
    1. fx.Graph treats copy_ as a side-effect node and does not erase it.
    2. Users may want to preserve some orphan `copy_` that is not from the
       optimizer.
    If the second reason does not hold, this pass can be rewritten as using
    DCE from fx.Graph (with the overwrite to the side-effect node list).
    """
    MAX_COPY_DISTANCE = 5
    remove_candidates: Set[fx.Node] = set()
    for node in reversed(gm.graph.nodes):
        if node.users:
            continue
        if node.op != OP.CALL_FUNCTION or node.target != aten.copy_.default:
            continue
        copy_ancestors: Set[fx.Node] = set()
        nodes = collections.deque([node, None])
        distance = 0
        should_remove = False
        while nodes and distance < MAX_COPY_DISTANCE:
            visiting = nodes.popleft()
            if visiting is None:
                distance += 1
                if nodes:
                    nodes.append(None)
                continue
            copy_ancestors.add(visiting)
            if visiting.op == OP.CALL_FUNCTION and str(visiting.target).startswith(('aten._foreach_', 'aten._fused_')):
                should_remove = True
            parents = pytree.arg_tree_leaves(*visiting.args, **visiting.kwargs)
            for parent in parents:
                if isinstance(parent, fx.Node):
                    nodes.append(parent)
        if should_remove:
            remove_candidates.update(copy_ancestors)
    for node in reversed(gm.graph.nodes):
        if node.users:
            continue
        if node not in remove_candidates:
            continue
        gm.graph.erase_node(node)