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
@dynamo_timed
def compile_to_module(self):
    from .codecache import PyCodeCache
    code, linemap = self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
    linemap = [(line_no, node.stack_trace) for line_no, node in linemap]
    key, path = PyCodeCache.write(code)
    mod = PyCodeCache.load_by_key_path(key, path, linemap=linemap, attrs=self.constants)
    self.cache_key = key
    self.cache_path = path
    self.cache_linemap = linemap
    assert mod.__file__ is not None
    log.debug('Output code written to: %s', mod.__file__)
    output_code_log.debug('Output code: \n%s', code)
    output_code_log.info('Output code written to: %s', mod.__file__)
    if config.benchmark_kernel:
        print(f'Compiled module path: {mod.__file__}', file=sys.stderr)
    V.debug.output_code(mod.__file__)
    V.debug.copy(os.path.splitext(mod.__file__)[0] + '.debug')
    return mod