import logging
import os
import tempfile
from enum import Enum
from typing import Callable, cast, Dict, Iterable, List, Set
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def is_leaf_subgraph(graph: fx.Graph, subgraph: List[fx.Node]) -> bool:
    """Ensure nodes in ``subgraph`` satisfy one of the following rules.

    1. The user of the node is in ``subgraph``.
    2. The user of the node is output.
    3. There are no users -- the node is a side-effect node.
    """
    all_nodes: Set[fx.Node] = set(subgraph)
    output = get_output(graph)
    for node in subgraph:
        for user in node.users:
            if not isinstance(user, fx.Node):
                continue
            if user not in all_nodes and user != output:
                return False
    return True