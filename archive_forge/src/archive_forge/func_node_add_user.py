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
def node_add_user(self, node: fx.Node, user: Any) -> None:
    for graph in self._all_graphs:
        actual_node = self._lookup_node(node, graph)
        if isinstance(user, fx.Node):
            actual_user_node = self._lookup_node(user, graph)
        else:
            actual_user_node = user
        assert actual_node is not None
        actual_node.users[actual_user_node] = None