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
def generate_extern_kernel_alloc_and_find_schema_if_needed_oss(self, name, kernel, codegen_args, cpp_op_schema, cpp_kernel_key, cpp_kernel_overload_name=''):
    if cpp_kernel_key not in self.extern_call_ops:
        self.writeline(f'static auto op_{cpp_kernel_key} = c10::Dispatcher::singleton()')
        self.writeline(f'\t.findSchemaOrThrow("{kernel}", "{cpp_kernel_overload_name}")')
        self.writeline(f'\t.typed<{cpp_op_schema}>();')
        self.extern_call_ops.add(cpp_kernel_key)
    self.writeline(f'auto {name} = op_{cpp_kernel_key}.call({', '.join(codegen_args)});')