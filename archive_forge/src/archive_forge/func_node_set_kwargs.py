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
def node_set_kwargs(self, node: fx.Node, kwargs: Dict[str, Argument]) -> None:
    if self._freeze_cross_iter_movement:
        node.kwargs = kwargs
        return
    setup_kwargs = tree_map_only(fx.Node, lambda _arg: self._lookup_node(_arg, self.setup_graph), kwargs)
    setup_node = self._lookup_node(node, self.setup_graph)
    assert setup_node is not None
    setup_node.kwargs = setup_kwargs
    cleanup_kwargs = tree_map_only(fx.Node, lambda _arg: self._lookup_node(_arg, self.cleanup_graph), kwargs)
    cleanup_node = self._lookup_node(node, self.cleanup_graph)
    assert cleanup_node is not None
    cleanup_node.kwargs = cleanup_kwargs
    node.kwargs = kwargs