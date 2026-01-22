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
def visit_Attribute(self, node):
    lhs = self.visit(node.value)
    while isinstance(lhs, ast.Attribute):
        lhs = self.visit(lhs.value)
    if lhs is None or (getattr(lhs, '__name__', '') == 'triton' or getattr(lhs, '__name__', '').endswith('.triton')):
        return None
    return getattr(lhs, node.attr)