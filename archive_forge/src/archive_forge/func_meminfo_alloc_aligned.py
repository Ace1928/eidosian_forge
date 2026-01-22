import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
@_check_null_result
def meminfo_alloc_aligned(self, builder, size, align):
    """
        Allocate a new MemInfo with an aligned data payload of `size` bytes.
        The data pointer is aligned to `align` bytes.  `align` can be either
        a Python int or a LLVM uint32 value.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    return self.meminfo_alloc_aligned_unchecked(builder, size, align)