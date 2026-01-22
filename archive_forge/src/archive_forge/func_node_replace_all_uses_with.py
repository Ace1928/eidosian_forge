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
def node_replace_all_uses_with(self, node: fx.Node, replace_with: fx.Node, delete_user_cb: Callable[[fx.Node], bool]=lambda user: True, *, propagate_meta=False) -> List[fx.Node]:
    for graph in self._all_graphs:
        actual_node = self._lookup_node(node, graph)
        actual_replace_with = self._lookup_node(replace_with, graph)
        assert actual_node is not None
        ret = actual_node.replace_all_uses_with(actual_replace_with, delete_user_cb, propagate_meta=propagate_meta)
    return ret