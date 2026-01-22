from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type
import torch
import torch.fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
from .virtualized import V
def is_aten_node(node):
    return node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload)