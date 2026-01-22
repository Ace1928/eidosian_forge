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
def gen_kernel(kernel):
    with contextlib.ExitStack() as stack:
        assert kernel
        if hasattr(kernel, 'codegen_inner_loops'):
            code.splice(kernel.preloads)
            kernel.codegen_inner_loops(code)
            stack.enter_context(code.indent())
        code.splice(kernel.loads)
        code.splice(kernel.compute)
        code.splice(kernel.stores)
    if hasattr(kernel, 'codegen_inner_loops'):
        code.splice(kernel.poststores)