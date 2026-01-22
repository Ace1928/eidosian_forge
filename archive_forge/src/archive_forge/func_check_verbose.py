import _collections_abc
import _weakrefset
import abc
import collections
import contextlib
import copy
import copyreg
import dataclasses
import enum
import functools
import importlib
import inspect
import linecache
import logging
import multiprocessing
import operator
import os
import posixpath
import random
import re
import selectors
import signal
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import weakref
from typing import Optional
import torch
import torch._inductor.test_operators
import torch.distributed
import torch.utils._content_store
from .utils import getfile
from .variables.functions import (
def check_verbose(obj, is_inlined_call=False):
    if isinstance(obj, (UserFunctionVariable, UserMethodVariable, NestedUserFunctionVariable)):
        try:
            py_obj = obj.get_function()
        except NotImplementedError:
            py_obj = None
        fi = FunctionInfo(py_obj, obj.get_name(), obj.get_filename(), obj.get_code())
    elif isinstance(obj, types.CodeType):
        fi = FunctionInfo(None, obj.co_name, obj.co_filename, obj)
    elif isinstance(obj, (types.FunctionType, types.MethodType)):
        fi = FunctionInfo(obj, obj.__name__, getfile(obj), obj.__code__)
    else:
        fi = FunctionInfo(obj, None, getfile(obj), None)
    if fi.code in get_func_inlinelist():
        return SkipResult(False, 'inlined according skipfiles.FUNC_INLINELIST')
    if is_inlined_call:
        if fi.name == 'patched_init':
            return SkipResult(True, 'patched init cannot be inlined.')
        elif fi.name == '__torch_function__':
            return SkipResult(False, 'allow inlining __torch_function__')
    return check_file(fi.filename, is_inlined_call)