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
def parseExpr(expr, module):
    try:
        value, len_parsed = parseNestedExpr(expr, module)
        assert len_parsed == len(expr), 'whole expression was not parsed, falling back to c++ parser'
        return value
    except Exception:
        '\n            The python resolver fails in several cases in known unit tests, and is intended\n            to fall back gracefully to the c++ resolver in general.  For example, python 2 style\n            annotations which are frequent in our unit tests often fail with types e.g. int not\n            resolvable from the calling frame.\n            '
        return None