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
def can_store_fp32_as_uint8(self, store_var: str, value_node: torch.fx.Node):
    """
        Check:
        1. store_type is torch.uint8
        2. value_node is of target to_dtype
        3. dtype of to_dtype node is torch.uint8
        """
    store_type = V.graph.get_dtype(store_var)
    if store_type not in [torch.uint8]:
        return False
    if value_node.target == 'to_dtype' and value_node.args[-1] == torch.uint8:
        return True
    return False