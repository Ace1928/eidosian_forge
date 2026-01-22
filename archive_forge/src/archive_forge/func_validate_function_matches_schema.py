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
def validate_function_matches_schema(schema: FunctionSchema, func: typing.Callable) -> None:
    sig = inspect.signature(func)
    if not all((supported_param(p) for _, p in sig.parameters.items())):
        raise ValueError(f'custom_op(..., manual_schema)(func): positional-only args, varargs, and kwargs are not supported. Please rewrite `func` to not have them. Got `func` with signature: {sig}')
    if any((p.annotation is not inspect.Parameter.empty for _, p in sig.parameters.items())) or sig.return_annotation is not inspect.Signature.empty:
        raise ValueError(f'custom_op(..., manual_schema)(func): When passing in a manual schema, we expect `func` to have no type annotations to avoid ambiguity. Got `func` with signature: {sig}')
    positional = [(name, param) for name, param in sig.parameters.items() if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    kwargonly = [(name, param) for name, param in sig.parameters.items() if param.kind == inspect.Parameter.KEYWORD_ONLY]

    def error():
        raise ValueError(f"custom_op(..., manual_schema)(func): When passing in a manual schema, we expect `func`'s signature to match `manual_schema` (aside from type annotations). func's signature: {sig}, manual_schema: {schema}")

    def error_default_args():
        raise ValueError(f"custom_op(..., manual_schema)(func): neither func nor manual_schema should have default arguments. Got func's signature: {sig}, manual_schema: {schema}")

    def compare(sig_args, schema_args):
        if len(sig_args) != len(schema_args):
            error()
        for (name, param), arg in zip(sig_args, schema_args):
            if name != arg.name:
                error()
            if param.default is not inspect.Parameter.empty or arg.default is not None:
                error_default_args()
    compare(positional, schema.arguments.flat_positional)
    compare(kwargonly, schema.arguments.flat_kwarg_only)