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
def get_annotation_str(annotation):
    """
    Convert an AST node containing a type annotation to the string present in the source
    that represents the same annotation.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return '.'.join([get_annotation_str(annotation.value), annotation.attr])
    elif isinstance(annotation, ast.Subscript):
        subscript_slice = annotation.slice if IS_PY39_PLUS else annotation.slice.value
        return f'{get_annotation_str(annotation.value)}[{get_annotation_str(subscript_slice)}]'
    elif isinstance(annotation, ast.Tuple):
        return ','.join([get_annotation_str(elt) for elt in annotation.elts])
    elif isinstance(annotation, (ast.Constant, ast.NameConstant)):
        return f'{annotation.value}'
    return None