import weakref
import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._ops import OpOverload
from torch.library import Library
from torchgen.model import (
from .autograd import autograd_not_implemented
def construct_functional_impl(mutable_op):

    def functional_impl(*args):
        new_args = []
        extra_rets = []
        for is_write, arg in zip(mutable_args(mutable_op), args):
            if is_write:
                cloned = arg.clone() if arg is not None else None
                new_args.append(cloned)
                extra_rets.append(cloned)
            else:
                new_args.append(arg)
        result = mutable_op(*new_args)
        if result is None:
            return tuple(extra_rets)
        if isinstance(result, tuple):
            return (*result, *extra_rets)
        return (result, *extra_rets)
    return functional_impl