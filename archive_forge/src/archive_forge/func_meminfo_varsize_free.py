import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_varsize_free(self, builder, meminfo, ptr):
    """
        Free a memory area allocated for a NRT varsize object.
        Note this does *not* free the NRT object itself!
        """
    self._require_nrt()
    mod = builder.module
    fnty = ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t, cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_MemInfo_varsize_free')
    return builder.call(fn, (meminfo, ptr))