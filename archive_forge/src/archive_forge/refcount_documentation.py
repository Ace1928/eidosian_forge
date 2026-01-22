from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import intrinsic
from numba.core.runtime.nrtdynmod import _meminfo_struct_type
Get the current refcount of an object.

    FIXME: only handles the first object
    