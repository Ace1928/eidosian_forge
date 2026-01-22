import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_data(self, builder, meminfo):
    """
        Given a MemInfo pointer, return a pointer to the allocated data
        managed by it.  This works for MemInfos allocated with all the
        above methods.
        """
    self._require_nrt()
    from numba.core.runtime.nrtdynmod import meminfo_data_ty
    mod = builder.module
    fn = cgutils.get_or_insert_function(mod, meminfo_data_ty, 'NRT_MemInfo_data_fast')
    return builder.call(fn, [meminfo])