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
@overload(random.randrange)
def randrange_impl_3(start, stop, step):
    if isinstance(start, types.Integer) and isinstance(stop, types.Integer) and isinstance(step, types.Integer):
        signed = max(start.signed, stop.signed, step.signed)
        bitwidth = max(start.bitwidth, stop.bitwidth, step.bitwidth)
        int_ty = types.Integer.from_bitwidth(bitwidth, signed)
        llvm_type = ir.IntType(bitwidth)
        start_preprocessor = _randrange_preprocessor(bitwidth, start)
        stop_preprocessor = _randrange_preprocessor(bitwidth, stop)
        step_preprocessor = _randrange_preprocessor(bitwidth, step)

        @intrinsic
        def _impl(typingcontext, start, stop, step):

            def codegen(context, builder, sig, args):
                start, stop, step = args
                start = start_preprocessor(builder, start, llvm_type)
                stop = stop_preprocessor(builder, stop, llvm_type)
                step = step_preprocessor(builder, step, llvm_type)
                return _randrange_impl(context, builder, start, stop, step, llvm_type, signed, 'py')
            return (signature(int_ty, start, stop, step), codegen)
        return lambda start, stop, step: _impl(start, stop, step)