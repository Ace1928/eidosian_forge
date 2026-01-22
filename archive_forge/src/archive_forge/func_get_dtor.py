import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
def get_dtor(self):
    """"Get the element dtor function pointer as void pointer.

        It's safe to be called multiple times.
        """
    dtor = self.define_dtor()
    dtor_fnptr = self._builder.bitcast(dtor, cgutils.voidptr_t)
    return dtor_fnptr