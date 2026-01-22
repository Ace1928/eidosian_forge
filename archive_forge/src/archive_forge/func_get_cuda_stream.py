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
def get_cuda_stream(idx=None):
    if idx is None:
        idx = get_current_device()
    try:
        from torch._C import _cuda_getCurrentRawStream
        return _cuda_getCurrentRawStream(idx)
    except ImportError:
        import torch
        return torch.cuda.current_stream(idx).cuda_stream