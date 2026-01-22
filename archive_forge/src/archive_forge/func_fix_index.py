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
def fix_index(self, idx):
    """
        Fix negative indices by adding the size to them.  Positive
        indices are left untouched.
        """
    is_negative = self._builder.icmp_signed('<', idx, ir.Constant(idx.type, 0))
    wrapped_index = self._builder.add(idx, self.size)
    return self._builder.select(is_negative, wrapped_index, idx)