from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type
import torch
import torch.fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
from .virtualized import V
def get_fake(x):
    if isinstance(x, torch.fx.Node):
        if 'val' not in x.meta:
            return x
        return x.meta['val']
    return x