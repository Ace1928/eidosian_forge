import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def divmod_by_constant(builder, val, divisor):
    """
    Compute the (quotient, remainder) of *val* divided by the constant
    positive *divisor*.  The semantics reflects those of Python integer
    floor division, rather than C's / LLVM's signed division and modulo.
    The difference lies with a negative *val*.
    """
    assert divisor > 0
    divisor = val.type(divisor)
    one = val.type(1)
    quot = alloca_once(builder, val.type)
    with builder.if_else(is_neg_int(builder, val)) as (if_neg, if_pos):
        with if_pos:
            quot_val = builder.sdiv(val, divisor)
            builder.store(quot_val, quot)
        with if_neg:
            val_plus_one = builder.add(val, one)
            quot_val = builder.sdiv(val_plus_one, divisor)
            builder.store(builder.sub(quot_val, one), quot)
    quot_val = builder.load(quot)
    rem_val = builder.sub(val, builder.mul(quot_val, divisor))
    return (quot_val, rem_val)