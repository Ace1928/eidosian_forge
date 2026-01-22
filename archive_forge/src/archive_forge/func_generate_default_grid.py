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
def generate_default_grid(self, name: str, grid: List[Any], cuda: bool=True):
    """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics.
        """
    if not cuda:
        return grid
    assert isinstance(grid, list), f'expected grid={grid!r} to be a list'
    grid = [e.inner_expr if isinstance(e, SymbolicCallArg) else e for e in grid]
    grid_fn = default_grid(*grid)
    params = CudaKernelParamCache.get(name)
    assert params is not None, f'cuda kernel parameters for {name} should already exist at this moment'
    block_cfg = {'XBLOCK': params['x_block'], 'YBLOCK': params['y_block'], 'ZBLOCK': params['z_block']}
    return grid_fn(block_cfg)