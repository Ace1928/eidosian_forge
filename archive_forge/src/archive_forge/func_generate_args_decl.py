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
def generate_args_decl(self, call_args):
    dynamic_symbols = V.graph.sizevars.free_symbols()
    new_args = []
    for arg in call_args:
        var_name = f'var_{next(self.arg_var_id)}'
        if isinstance(arg, (sympy.Integer, sympy.Symbol, SymbolicCallArg)):
            self.writeline(f'auto {var_name} = {arg};')
        elif isinstance(arg, sympy.Expr):
            self.writeline(f'auto {var_name} = {self.expr_printer(arg)};')
        elif is_int(arg):
            self.writeline(f'int {var_name} = {arg};')
        elif is_float(arg):
            self.writeline(f'float {var_name} = {arg};')
        elif any((str(arg) == s.name for s in dynamic_symbols)):
            self.writeline(f'auto {var_name} = {arg};')
        elif arg == 'nullptr':
            self.writeline(f'auto {var_name} = nullptr;')
        elif arg == 'c10::nullopt':
            self.writeline(f'auto {var_name} = c10::nullopt;')
        elif config.aot_inductor.abi_compatible:
            self.writeline(f'CUdeviceptr {var_name};')
            self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr({arg}, reinterpret_cast<void**>(&{var_name})));')
        else:
            self.writeline(f'CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({arg}.data_ptr());')
        new_args.append(f'&{var_name}')
    return ', '.join(new_args)