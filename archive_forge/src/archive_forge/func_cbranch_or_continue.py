import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def cbranch_or_continue(builder, cond, bbtrue):
    """
    Branch conditionally or continue.

    Note: a new block is created and builder is moved to the end of the new
          block.
    """
    bbcont = builder.append_basic_block('.continue')
    builder.cbranch(cond, bbtrue, bbcont)
    builder.position_at_end(bbcont)
    return bbcont