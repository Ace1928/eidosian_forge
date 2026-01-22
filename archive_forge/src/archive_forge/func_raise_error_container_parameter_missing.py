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
def raise_error_container_parameter_missing(target_type) -> None:
    if target_type == 'Dict':
        raise RuntimeError('Attempted to use Dict without contained types. Please add contained type, e.g. Dict[int, int]')
    raise RuntimeError(f'Attempted to use {target_type} without a contained type. Please add a contained type, e.g. {target_type}[int]')