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
def get_abstract_impl(qualname):
    if qualname not in torch._custom_op.impl.global_registry:
        return None
    custom_op = torch._custom_op.impl.global_registry[qualname]
    if custom_op is None:
        return None
    if not custom_op._has_impl('abstract'):
        return None
    return custom_op._get_impl('abstract').func