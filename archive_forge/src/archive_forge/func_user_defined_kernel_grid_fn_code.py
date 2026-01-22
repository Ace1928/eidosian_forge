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
def user_defined_kernel_grid_fn_code(name, configs, grids):
    output = IndentedBuffer()
    fn_name = f'grid_wrapper_for_{name}'
    output.writeline(f'def {fn_name}(meta):')
    with output.indent():
        if len(grids) == 1:
            output.writeline(f'return {grids[0]}')
        else:
            assert len(grids) == len(configs)
            for grid, c in zip(grids, configs):
                guards = [f"meta['{name}'] == {val}" for name, val in c.kwargs.items()]
                guards = ' and '.join(guards)
                output.writeline(f'if {guards}: return {grid}')
    return (fn_name, output.getvalue())