import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def for_range(builder, count, start=None, intp=None):
    """
    Generate LLVM IR for a for-loop in [start, count).
    *start* is equal to 0 by default.

    Yields a Loop namedtuple with the following members:
    - `index` is the loop index's value
    - `do_break` is a no-argument callable to break out of the loop
    """
    if intp is None:
        intp = count.type
    if start is None:
        start = intp(0)
    stop = count
    bbcond = builder.append_basic_block('for.cond')
    bbbody = builder.append_basic_block('for.body')
    bbend = builder.append_basic_block('for.end')

    def do_break():
        builder.branch(bbend)
    bbstart = builder.basic_block
    builder.branch(bbcond)
    with builder.goto_block(bbcond):
        index = builder.phi(intp, name='loop.index')
        pred = builder.icmp_signed('<', index, stop)
        builder.cbranch(pred, bbbody, bbend)
    with builder.goto_block(bbbody):
        yield Loop(index, do_break)
        bbbody = builder.basic_block
        incr = increment_index(builder, index)
        terminate(builder, bbcond)
    index.add_incoming(start, bbstart)
    index.add_incoming(incr, bbbody)
    builder.position_at_end(bbend)