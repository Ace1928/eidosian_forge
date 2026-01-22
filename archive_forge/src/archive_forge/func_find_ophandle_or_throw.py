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
def find_ophandle_or_throw(cpp_ns: str, operator_name: OperatorName):
    overload_name = '' if operator_name.overload_name is None else operator_name.overload_name
    return _C._dispatch_find_schema_or_throw(f'{cpp_ns}::{str(operator_name.name)}', overload_name)