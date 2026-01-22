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
def fill_output_arg(arg, return_type):
    if isinstance(return_type, torch.TensorType):
        self.writeline(f'AtenTensorHandle {arg}_handle;  // output buffer')
        self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{arg}_handle));')
        self.writeline(f'RAIIAtenTensorHandle {arg}({arg}_handle);')
        new_tensor_args.append(f'{arg}')
    elif isinstance(return_type, torch.SymIntType):
        raise NotImplementedError('NYI support for return type: SymInt')
    elif isinstance(return_type, torch.ListType) and isinstance(return_type.getElementType(), torch.SymIntType):
        raise NotImplementedError('NYI support for return type: List[SymInt]')
    else:
        raise AssertionError(f'Unsupported return type found: {return_type}')