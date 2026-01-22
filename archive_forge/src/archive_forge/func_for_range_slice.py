import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def for_range_slice(builder, start, stop, step, intp=None, inc=True):
    """
    Generate LLVM IR for a for-loop based on a slice.  Yields a
    (index, count) tuple where `index` is the slice index's value
    inside the loop, and `count` the iteration count.

    Parameters
    -------------
    builder : object
        IRBuilder object
    start : int
        The beginning value of the slice
    stop : int
        The end value of the slice
    step : int
        The step value of the slice
    intp :
        The data type
    inc : boolean, optional
        Signals whether the step is positive (True) or negative (False).

    Returns
    -----------
        None
    """
    if intp is None:
        intp = start.type
    bbcond = builder.append_basic_block('for.cond')
    bbbody = builder.append_basic_block('for.body')
    bbend = builder.append_basic_block('for.end')
    bbstart = builder.basic_block
    builder.branch(bbcond)
    with builder.goto_block(bbcond):
        index = builder.phi(intp, name='loop.index')
        count = builder.phi(intp, name='loop.count')
        if inc:
            pred = builder.icmp_signed('<', index, stop)
        else:
            pred = builder.icmp_signed('>', index, stop)
        builder.cbranch(pred, bbbody, bbend)
    with builder.goto_block(bbbody):
        yield (index, count)
        bbbody = builder.basic_block
        incr = builder.add(index, step)
        next_count = increment_index(builder, count)
        terminate(builder, bbcond)
    index.add_incoming(start, bbstart)
    index.add_incoming(incr, bbbody)
    count.add_incoming(ir.Constant(intp, 0), bbstart)
    count.add_incoming(next_count, bbbody)
    builder.position_at_end(bbend)