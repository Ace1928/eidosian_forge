import dataclasses
import functools
import inspect
import sys
import typing
import weakref
from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy
import torch
import torch._C as _C
import torch.library as library
from torch._library.abstract_impl import AbstractImplCtx
from torch.library import get_ctx
from .autograd import autograd_kernel_indirection, construct_autograd_kernel
def report_error_callback(custom_op: typing.Any, key: str) -> None:
    if key == 'Undefined':
        raise NotImplementedError(f'{custom_op}: There were no Tensor inputs to this operator (e.g. you passed an empty list of Tensors). If your operator is a factory function (that is, it takes no Tensors and constructs a new one), then please use CustomOp.impl_factory to register an implementation for it')
    if key == 'Meta':
        raise NotImplementedError(f"{custom_op}: when running with device='Meta' tensors: there is no abstract impl registered for this CustomOp. Please register one via CustomOp.impl_abstract to get this CustomOp to work with Meta tensors")
    if key in ('CPU', 'CUDA'):
        device = key.lower()
        raise NotImplementedError(f"{custom_op}: when running with device='{device}' tensors: there is no {device} impl registered for this CustomOp. Please register one via CustomOp.impl(device_type='{device}')")
    raise NotImplementedError(f"{custom_op}: No implementation for dispatch key {key}. It is likely that we have not added this functionality yet, please either open an issue or if you're feeling adventurous, use the low-level torch.library API")