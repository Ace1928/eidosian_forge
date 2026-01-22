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
def codegen_inputs(self, code: IndentedBuffer, graph_inputs: Dict[str, ir.TensorBox]):
    """Assign all symbolic shapes to locals"""

    @functools.lru_cache(None)
    def sizeof(name):
        self.codegen_input_size_var_decl(code, name)
        return f'{name}_size'

    @functools.lru_cache(None)
    def strideof(name):
        self.codegen_input_stride_var_decl(code, name)
        return f'{name}_stride'
    needed = V.graph.sizevars.free_symbols()

    def is_expr(x):
        return isinstance(x[1], sympy.Expr)
    graph_inputs_expr = list(filter(is_expr, graph_inputs.items()))
    graph_inputs_tensors = list(filter(lambda x: not is_expr(x), graph_inputs.items()))
    for name, shape in graph_inputs_expr:
        shape = V.graph.sizevars.simplify(shape)
        if shape in needed:
            needed.remove(shape)
            code.writeline(f'{self.declare}{shape} = {name}{self.ending}')
    for name, value in graph_inputs_tensors:
        shapes = value.get_size()
        for dim, shape in enumerate(shapes):
            shape = V.graph.sizevars.simplify(shape)
            if shape in needed:
                needed.remove(shape)
                code.writeline(f'{self.declare}{shape} = {sizeof(name)}[{dim}]{self.ending}')
    for name, value in graph_inputs_tensors:
        shapes = value.get_stride()
        for dim, shape in enumerate(shapes):
            shape = V.graph.sizevars.simplify(shape)
            if shape in needed:
                needed.remove(shape)
                code.writeline(f'{self.declare}{shape} = {strideof(name)}[{dim}]{self.ending}')