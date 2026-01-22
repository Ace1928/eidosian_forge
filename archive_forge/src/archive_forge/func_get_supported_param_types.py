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
def get_supported_param_types():
    data = [(torch.Tensor, 'Tensor', True, True, False), (int, 'SymInt', True, False, True), (float, 'float', True, False, True), (bool, 'bool', True, False, True), (str, 'str', False, False, False), (torch.types.Number, 'Scalar', True, False, False), (torch.dtype, 'ScalarType', False, False, False), (torch.device, 'Device', False, False, False)]
    result = []
    for line in data:
        result.extend(derived_types(*line))
    return dict(result)