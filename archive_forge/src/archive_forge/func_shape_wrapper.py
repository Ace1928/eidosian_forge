import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def shape_wrapper(f):

    @wraps(f)
    def nf(*args, out=None, **kwargs):
        args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out))
        return f(*args, out_shape=out_shape, **kwargs)
    return nf