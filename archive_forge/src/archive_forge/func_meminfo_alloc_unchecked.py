import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_alloc_unchecked(self, builder, size):
    """
        Allocate a new MemInfo with a data payload of `size` bytes.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    self._require_nrt()
    mod = builder.module
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
    fn = cgutils.get_or_insert_function(mod, fnty, self._meminfo_api.alloc)
    fn.return_value.add_attribute('noalias')
    return builder.call(fn, [size])