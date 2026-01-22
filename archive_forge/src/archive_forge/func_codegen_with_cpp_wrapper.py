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
def codegen_with_cpp_wrapper(self):
    """
        For CPU, the cpp wrapper codegen is done in one pass.
        For GPU, the cpp wrapper codegen is done in two steps: JIT-compile the model with python
        wrapper code and run it to generate autotuned kernel binaries in the first pass; and then
        generate cpp wrapper code and compile it to a dynamic library in the second pass.
        """
    if 'cuda' in self.device_types:
        self.cpp_wrapper = False
        compiled = self.compile_to_module().call

        def materialize(x):
            if isinstance(x, (torch.SymInt, torch.SymFloat)):
                return x.node.hint
            elif isinstance(x, FakeTensor):
                return defake(x)
            else:
                assert isinstance(x, torch.Tensor), 'Unknown type when creating real inputs' + str(type(x))
                return x
        with torch.utils._python_dispatch._disable_current_modes():
            assert self.example_inputs is not None
            real_inputs = [materialize(x) for x in self.example_inputs]
            compiled(real_inputs)
        del real_inputs
        self.cpp_wrapper = True
        self.removed_buffers.clear()
        self.inplaced_to_remove.clear()
        return self.codegen()
    else:
        return self.codegen()