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
@overload(random.getrandbits)
def getrandbits_impl(k):
    if isinstance(k, types.Integer):

        @intrinsic
        def _impl(typingcontext, k):

            def codegen(context, builder, sig, args):
                nbits, = args
                too_large = builder.icmp_unsigned('>=', nbits, const_int(65))
                too_small = builder.icmp_unsigned('==', nbits, const_int(0))
                with cgutils.if_unlikely(builder, builder.or_(too_large, too_small)):
                    msg = 'getrandbits() limited to 64 bits'
                    context.call_conv.return_user_exc(builder, OverflowError, (msg,))
                state_ptr = get_state_ptr(context, builder, 'py')
                return get_next_int(context, builder, state_ptr, nbits, False)
            return (signature(types.uint64, k), codegen)
        return lambda k: _impl(k)