import hashlib
import logging
import operator
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple
import sympy
import torch
import torch._logging
import torch.fx
from torch._decomp import get_decompositions
from torch._dynamo.utils import defake, dynamo_timed
from torch._logging import LazyString
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import has_free_symbols, ShapeEnv, SymTypes
from torch.utils._mode_utils import no_dispatch
from . import config, ir
from .codegen.common import (
from .codegen.wrapper import CppWrapperCodeGen, CudaWrapperCodeGen, WrapperCodeGen
from .exc import (
from .ir import (
from .lowering import (
from .sizevars import SizeVarAllocator
from .utils import convert_shape_to_inductor, gather_origins, get_sympy_Expr_dtype
from .virtualized import V
def validate_can_generate_cpp_wrapper(self):
    if config.disable_cpp_codegen:
        raise CppWrapperCodeGenError('C++ codegen is disabled')
    if sys.platform != 'linux':
        raise CppWrapperCodeGenError(f'Unsupported platform {sys.platform}')
    for value in self.graph_inputs.values():
        dtype = None
        if isinstance(value, TensorBox):
            dtype = value.get_dtype()
        elif isinstance(value, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)):
            dtype = may_get_constant_buffer_dtype(value)
        if not supported_dtype_of_cpp_wrapper(dtype, self.cuda):
            raise CppWrapperCodeGenError(f'Unsupported input dtype {dtype}')