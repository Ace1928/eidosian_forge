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
def get_overload_no_implementation_error_message(kind, obj):
    sourcelines, file_lineno, filename = get_source_lines_and_file(obj)
    return f'Implementation for the {kind} "{_qualified_name(obj)}" is missing. Please make sure a definition is provided and defined after all overload declarations.\nFile "{filename}", line {file_lineno}:\n' + ''.join(sourcelines) + '\n' + _OVERLOAD_EXAMPLE