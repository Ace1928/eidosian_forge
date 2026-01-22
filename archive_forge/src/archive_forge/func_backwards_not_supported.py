import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def backwards_not_supported(prim):

    def redispatch_prim(args, kwargs):
        with torch._C._AutoDispatchBelowAutograd():
            old = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView)
            return prim(*args, **kwargs)

    class BackwardsNotSupported(torch.autograd.Function):

        @staticmethod
        def forward(ctx, args_spec, *flat_args):
            args, kwargs = tree_unflatten(flat_args, args_spec)
            return redispatch_prim(args, kwargs)

        @staticmethod
        def backward(ctx, *args):
            raise RuntimeError('backwards not supported on prim')

    @wraps(prim)
    def _autograd_impl(*args, **kwargs):
        flat_args, args_spec = tree_flatten((args, kwargs))
        if torch.is_grad_enabled() and any((a.requires_grad for a in flat_args if isinstance(a, torch.Tensor))):
            return BackwardsNotSupported.apply(args_spec, *flat_args)
        else:
            return redispatch_prim(args, kwargs)
    return _autograd_impl