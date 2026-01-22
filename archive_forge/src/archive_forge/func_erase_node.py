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
def erase_node(self, to_erase: fx.Node) -> None:
    if self._freeze_cross_iter_movement:
        return super().erase_node(to_erase)
    setup_node = self._lookup_node(to_erase, self.setup_graph)
    assert setup_node is not None, 'setup_node is None'
    self.setup_graph.erase_node(setup_node)
    super().erase_node(to_erase)
    cleanup_node = self._lookup_node(to_erase, self.cleanup_graph)
    self.cleanup_graph.erase_node(cleanup_node)