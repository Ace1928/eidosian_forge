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
def find_all_descendants(gm: IterGraphModule, parent_nodes: List[fx.Node]) -> List[fx.Node]:
    """Identify the list of nodes to move during FX graph transformation."""
    assert len(parent_nodes) > 0, 'No parent nodes are given.'
    output = get_output(gm.graph)
    dq_parent_nodes = collections.deque(parent_nodes)
    move_node_set = set()
    while dq_parent_nodes:
        node = dq_parent_nodes.popleft()
        move_node_set.add(node)
        dq_parent_nodes += [u for u in node.users if isinstance(u, fx.Node) and u != output]
    move_nodes = [node for node in gm.graph.nodes if node in move_node_set]
    return move_nodes