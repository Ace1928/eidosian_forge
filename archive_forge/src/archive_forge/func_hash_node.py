from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type
import torch
import torch.fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
from .virtualized import V
def hash_node(self, node: torch.fx.Node):
    return (node, node.target, id(node.args), id(node.kwargs))