import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def wrap_index_literal(idx, size):
    if idx < 0:
        if idx <= -size:
            return 0
        else:
            return idx + size
    elif idx >= size:
        return size
    else:
        return idx