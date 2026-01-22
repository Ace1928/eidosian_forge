from collections import namedtuple
from copy import deepcopy
from itertools import combinations
import torch
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def has_aliased(lhs, rhs):
    try:
        return torch._C._overlaps(lhs, rhs)
    except Exception as exception:
        if str(exception).startswith('Cannot inspect value of type '):
            return False
        else:
            raise exception