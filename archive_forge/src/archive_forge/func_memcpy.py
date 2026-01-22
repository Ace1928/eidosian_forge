import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def memcpy(builder, dst, src, count):
    """
    Emit a memcpy to the builder.

    Copies each element of dst to src. Unlike the C equivalent, each element
    can be any LLVM type.

    Assumes
    -------
    * dst.type == src.type
    * count is positive
    """
    assert dst.type == src.type
    with for_range(builder, count, intp=count.type) as loop:
        out_ptr = builder.gep(dst, [loop.index])
        in_ptr = builder.gep(src, [loop.index])
        builder.store(builder.load(in_ptr), out_ptr)