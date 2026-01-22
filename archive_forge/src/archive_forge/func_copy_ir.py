import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def copy_ir(the_ir):
    kernel_copy = the_ir.copy()
    kernel_copy.blocks = {}
    for block_label, block in the_ir.blocks.items():
        new_block = copy.deepcopy(the_ir.blocks[block_label])
        kernel_copy.blocks[block_label] = new_block
    return kernel_copy