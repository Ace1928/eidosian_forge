import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
@_check_null_result
def meminfo_alloc_dtor(self, builder, size, dtor):
    """
        Allocate a new MemInfo with a data payload of `size` bytes and a
        destructor `dtor`.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    return self.meminfo_alloc_dtor_unchecked(builder, size, dtor)