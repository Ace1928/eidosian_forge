import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def node_append(self, target_node: fx.Node, node: fx.Node) -> None:
    """Append node to target_node."""
    if self._freeze_cross_iter_movement:
        target_node.append(node)
        return
    for graph in self._all_graphs:
        actual_node = self._lookup_node(node, graph)
        assert actual_node is not None, f'The actual node is None, {node}.'
        actual_target_node = self._lookup_node(target_node, graph)
        assert actual_target_node is not None, f'The actual target node is None, {target_node}.'
        actual_target_node.append(actual_node)