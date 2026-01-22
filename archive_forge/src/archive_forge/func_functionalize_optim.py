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
def functionalize_optim(self) -> None:
    for node in reversed(self.nodes):
        if node.name.startswith('output'):
            output_node = node
        elif node.name.startswith('_fused_adam_'):
            optim_node = node
        elif node.name.startswith('_foreach_add_'):
            step_node = node
            self.node_add_user(optim_node, output_node)
            self.node_add_user(step_node, optim_node)