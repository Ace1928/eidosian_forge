import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
def convert_arg_type(python_type: str):
    from .cpp import CONTAINER_PYTHON_TO_CPP, PYTHON_TO_CPP
    if python_type == 'Tensor':
        return f'at::{python_type} const&'
    if python_type in PYTHON_TO_CPP:
        return PYTHON_TO_CPP[python_type]
    for py_container, cpp_container in CONTAINER_PYTHON_TO_CPP.items():
        container_match = re.findall(py_container + '\\[([a-zA-Z_]+)]', python_type)
        if len(container_match) == 1:
            contained_type = container_match[0]
            assert contained_type in PYTHON_TO_CPP, f'unsupported {py_container} type in convert_arg_type: {contained_type}'
            cpp_contained_type = PYTHON_TO_CPP[contained_type]
            return f'{cpp_container}<{cpp_contained_type}>'
    raise AssertionError(f'unsupport python_type: {python_type}')