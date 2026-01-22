import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
@classmethod
def from_set(cls, context, builder, iter_type, set_val):
    set_inst = SetInstance(context, builder, iter_type.container, set_val)
    self = cls(context, builder, iter_type, None)
    index = context.get_constant(types.intp, 0)
    self._iter.index = cgutils.alloca_once_value(builder, index)
    self._iter.meminfo = set_inst.meminfo
    return self