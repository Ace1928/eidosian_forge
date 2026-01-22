import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import types
import typing
import warnings
import weakref
from textwrap import dedent
from typing import (  # noqa: F401
import torch
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future
def get_callable_argument_names(fn) -> List[str]:
    """
    Gets names of all POSITIONAL_OR_KEYWORD arguments for callable `fn`.
    Returns an empty list when other types of arguments are present.

    This is used by `torch.jit.trace` to assign meaningful argument names to
    traced functions and modules.

    Args:
        fn: A callable.
    Returns:
        Argument names: List[str]
    """
    try:
        callable_signature = inspect.signature(fn)
    except Exception:
        return []
    argument_names = []
    for name, param in callable_signature.parameters.items():
        if not param.kind == param.POSITIONAL_OR_KEYWORD:
            continue
        argument_names.append(name)
    return argument_names