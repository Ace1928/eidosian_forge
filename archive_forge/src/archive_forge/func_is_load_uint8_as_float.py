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
def is_load_uint8_as_float(self, name: str, users: Dict[torch.fx.Node, None]):
    """
        Check:
        1. load_type is torch.uint8
        2. has 1 user node of target to_dtype
        3. dtype of to_dtype is torch.float
        """
    load_type = V.graph.get_dtype(name)
    if load_type is not torch.uint8:
        return False
    if len(users) == 1:
        user = next(iter(users))
        if user.target == 'to_dtype' and user.args[-1] == torch.float:
            return True
        return False
    return False