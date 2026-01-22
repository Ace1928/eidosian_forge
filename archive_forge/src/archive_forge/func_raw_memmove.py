import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def raw_memmove(builder, dst, src, count, itemsize, align=1):
    """
    Emit a raw memmove() call for `count` items of size `itemsize`
    from `src` to `dest`.
    """
    return _raw_memcpy(builder, 'llvm.memmove', dst, src, count, itemsize, align)