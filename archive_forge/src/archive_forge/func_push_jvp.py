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
def push_jvp(basis):
    output = _jvp_with_argnums(func, args, basis, argnums=argnums, has_aux=has_aux)
    error_if_complex('jacfwd', output[0], is_input=False)
    if has_aux:
        _, jvp_out, aux = output
        return (jvp_out, aux)
    _, jvp_out = output
    return jvp_out