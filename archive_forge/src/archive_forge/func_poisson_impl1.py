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
@overload(np.random.poisson)
def poisson_impl1(lam):
    if isinstance(lam, (types.Float, types.Integer)):

        @intrinsic
        def _impl(typingcontext, lam):
            lam_preprocessor = _double_preprocessor(lam)

            def codegen(context, builder, sig, args):
                state_ptr = get_np_state_ptr(context, builder)
                retptr = cgutils.alloca_once(builder, int64_t, name='ret')
                bbcont = builder.append_basic_block('bbcont')
                bbend = builder.append_basic_block('bbend')
                lam, = args
                lam = lam_preprocessor(builder, lam)
                big_lam = builder.fcmp_ordered('>=', lam, ir.Constant(double, 10.0))
                with builder.if_then(big_lam):
                    fnty = ir.FunctionType(int64_t, (rnd_state_ptr_t, double))
                    fn = cgutils.get_or_insert_function(builder.function.module, fnty, 'numba_poisson_ptrs')
                    ret = builder.call(fn, (state_ptr, lam))
                    builder.store(ret, retptr)
                    builder.branch(bbend)
                builder.branch(bbcont)
                builder.position_at_end(bbcont)
                _random = np.random.random
                _exp = math.exp

                def poisson_impl(lam):
                    """Numpy's algorithm for poisson() on small *lam*.

                    This method is invoked only if the parameter lambda of the
                    distribution is small ( < 10 ). The algorithm used is
                    described in "Knuth, D. 1969. 'Seminumerical Algorithms.
                    The Art of Computer Programming' vol 2.
                    """
                    if lam < 0.0:
                        raise ValueError('poisson(): lambda < 0')
                    if lam == 0.0:
                        return 0
                    enlam = _exp(-lam)
                    X = 0
                    prod = 1.0
                    while 1:
                        U = _random()
                        prod *= U
                        if prod <= enlam:
                            return X
                        X += 1
                ret = context.compile_internal(builder, poisson_impl, sig, args)
                builder.store(ret, retptr)
                builder.branch(bbend)
                builder.position_at_end(bbend)
                return builder.load(retptr)
            return (signature(types.int64, lam), codegen)
        return lambda lam: _impl(lam)