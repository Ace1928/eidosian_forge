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
def compute_expected_grad(args, tangent_args, kwargs, tangent_kwargs):
    op_args = tuple(map(maybe_make_dual, zip(args, tangent_args)))
    op_kwargs = {k: maybe_make_dual((v, tangent_kwargs[k])) for k, v in kwargs.items()}
    if gradcheck_wrapper is None:
        return op(*op_args, **op_kwargs)
    return gradcheck_wrapper(op, *op_args, **op_kwargs)