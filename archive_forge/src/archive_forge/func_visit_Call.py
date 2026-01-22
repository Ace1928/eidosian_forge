from __future__ import annotations, division
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
def visit_Call(self, node):
    func = self.visit(node.func)
    if func is None:
        return
    if inspect.isbuiltin(func):
        return
    if func.__module__ and (func.__module__.startswith('triton.') or '.triton.' in func.__module__):
        return
    assert isinstance(func, JITFunction), f'Function "{func.__name__}" is being called from a Triton function but is not a Triton function itself. Decorate it with @triton.jit to fix this'
    func_cache_key = func.cache_key
    noinline = str(getattr(func, 'noinline', False))
    self.ret = (self.ret + func_cache_key + noinline).encode('utf-8')
    self.ret = hashlib.sha1(self.ret).hexdigest()