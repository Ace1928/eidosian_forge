from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
def error_if_complex(func_name, args, is_input):
    flat_args = pytree.tree_leaves(args)
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor) and arg.dtype.is_complex:
            input_or_output = 'inputs' if is_input else 'outputs'
            err_msg = f'{func_name}: Expected all {input_or_output} to be real but received complex tensor at flattened input idx: {idx}'
            raise RuntimeError(err_msg)