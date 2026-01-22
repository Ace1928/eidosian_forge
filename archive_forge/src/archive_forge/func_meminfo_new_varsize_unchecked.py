import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_new_varsize_unchecked(self, builder, size):
    """
        Allocate a MemInfo pointing to a variable-sized data area.  The area
        is separately allocated (i.e. two allocations are made) so that
        re-allocating it doesn't change the MemInfo's address.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    self._require_nrt()
    mod = builder.module
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
    fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_MemInfo_new_varsize')
    fn.return_value.add_attribute('noalias')
    return builder.call(fn, [size])