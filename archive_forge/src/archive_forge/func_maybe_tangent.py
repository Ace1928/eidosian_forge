import torch
from torch import Tensor
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from typing import Callable
import re
def maybe_tangent(t):
    assert type(t) is not CCT
    if isinstance(t, torch.Tensor) and t.requires_grad:
        return torch.randn_like(t)
    elif is_tensorlist(t):
        return [torch.randn_like(e) if e.requires_grad else None for e in t]
    return None