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
def do_split_with_tiling():
    sympy_factor = sympy.Integer(factor)
    offset = FloorDiv(self.size, sympy_factor) * sympy_factor
    main_loop = LoopLevel(self.var, offset)
    main_loop.steps = sympy_factor
    main_loop.parallel = self.parallel
    main_loop.collapsed = False
    main_loop.reduction_var_map = self.reduction_var_map
    main_loop.inner = clone_inner()
    if main_loop.inner:
        for loop in main_loop.inner:
            loop.parent = main_loop
    tail_loop = LoopLevel(self.var, self.size)
    tail_loop.offset = offset
    tail_loop.parallel = self.parallel
    tail_loop.collapsed = False
    tail_loop.reduction_var_map = self.reduction_var_map
    tail_loop.inner = clone_inner()
    if tail_loop.inner:
        for loop in tail_loop.inner:
            loop.parent = tail_loop
    return (main_loop, tail_loop)