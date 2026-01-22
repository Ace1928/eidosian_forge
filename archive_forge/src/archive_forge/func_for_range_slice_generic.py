import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def for_range_slice_generic(builder, start, stop, step):
    """
    A helper wrapper for for_range_slice().  This is a context manager which
    yields two for_range_slice()-alike context managers, the first for
    the positive step case, the second for the negative step case.

    Use:
        with for_range_slice_generic(...) as (pos_range, neg_range):
            with pos_range as (idx, count):
                ...
            with neg_range as (idx, count):
                ...
    """
    intp = start.type
    is_pos_step = builder.icmp_signed('>=', step, ir.Constant(intp, 0))
    pos_for_range = for_range_slice(builder, start, stop, step, intp, inc=True)
    neg_for_range = for_range_slice(builder, start, stop, step, intp, inc=False)

    @contextmanager
    def cm_cond(cond, inner_cm):
        with cond:
            with inner_cm as value:
                yield value
    with builder.if_else(is_pos_step, likely=True) as (then, otherwise):
        yield (cm_cond(then, pos_for_range), cm_cond(otherwise, neg_for_range))