import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
def get_rnd_shuffle(builder):
    """
    Get the internal function to shuffle the MT taste.
    """
    fnty = ir.FunctionType(ir.VoidType(), (rnd_state_ptr_t,))
    fn = cgutils.get_or_insert_function(builder.function.module, fnty, 'numba_rnd_shuffle')
    fn.args[0].add_attribute('nocapture')
    return fn