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
def is_load_integer_scalar_tensor(self, name: str, index: sympy.Expr):
    load_dtype = V.graph.get_dtype(name)
    buffer = V.graph.get_buffer(name)
    return load_dtype in [torch.int32, torch.int64] and isinstance(buffer, TensorBox) and isinstance(buffer.data, StorageBox) and (len(buffer.data.layout.size) == 0) and (index == 0)