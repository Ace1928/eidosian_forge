import math
import sys
import itertools
from collections import namedtuple
import llvmlite.ir as ir
import numpy as np
import operator
from numba.np import arrayobj, ufunc_db, numpy_support
from numba.core.imputils import Registry, impl_ret_new_ref, force_error_model
from numba.core import typing, types, utils, cgutils, callconv
from numba.np.numpy_support import (
from numba.core.typing import npydecl
from numba.core.extending import overload, intrinsic
from numba.core import errors
from numba.cpython import builtins
def update_indices(self, loop_indices, name):
    bld = self.array.builder
    intpty = self.array.context.get_value_type(types.intp)
    ONE = ir.Constant(ir.IntType(intpty.width), 1)
    indices = loop_indices[len(loop_indices) - len(self.indices):]
    for src, dst, dim in zip(indices, self.indices, self.array.shape):
        cond = bld.icmp_unsigned('>', dim, ONE)
        with bld.if_then(cond):
            bld.store(src, dst)