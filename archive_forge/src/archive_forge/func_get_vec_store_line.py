import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
def get_vec_store_line(self, value, var, index, dtype):
    """
        Get a store line str that stores `value` into `var` at `index` of `dtype`.
        :param value: Vectorized type templaterized on `dtype`.
        :param var: buffer to store into.
        :index: index into the `var`.
        """
    assert isinstance(value, str) or (isinstance(value, CppCSEVariable) and value.is_vec), value
    tiling_var = self.itervars[self.tiling_idx]
    assert index.has(tiling_var)
    var_expr = f'{var} + {cexpr_index(index)}'
    non_contiguous = stride_at(tiling_var, index) != 1 or 'tmp' in f'{index}'
    if non_contiguous:
        var_expr = 'tmpbuf'
    if dtype == torch.float:
        line = f'{value}.store({var_expr});'
    else:
        line = f'{value}.store({var_expr}, {self.tiling_factor});'
    if non_contiguous:
        inner = sympy_symbol(f'{tiling_var}_inner')
        new_index = self.scale_index_with_offset(index, itervar_idx=self.tiling_idx, offset=inner)
        tmp_bufsize = f'{self.tiling_factor}*sizeof(float)/sizeof({DTYPE_TO_CPP[dtype]})'
        line = f'{{ __at_align__ {DTYPE_TO_CPP[dtype]} tmpbuf[{tmp_bufsize}]; {line} for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++) {var}[{cexpr_index(new_index)}] = tmpbuf[{inner}]; }}'
    return line