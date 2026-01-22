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
def forward_ad_checks(flat_tangents):
    for idx, t in enumerate(flat_tangents):
        if t.shape != flat_primals_shape[idx]:
            msg = f"tangent:{idx} with shape {t.shape} in flattened pytree doesn't match the shape {flat_primals_shape[idx]} of the corresponding primal."
            raise RuntimeError(msg)
        if t.device != flat_primals_device[idx]:
            msg = f"tangent:{idx} with device {t.device} in flattened pytree doesn't match the device {flat_primals_device[idx]} of the corresponding primal."
            raise RuntimeError(msg)
        if t.dtype != flat_primals_dtype[idx]:
            msg = f"tangent:{idx} with dtype {t.dtype} in flattened pytree doesn't match the dtype {flat_primals_dtype[idx]} of the corresponding primal."
            raise RuntimeError(msg)