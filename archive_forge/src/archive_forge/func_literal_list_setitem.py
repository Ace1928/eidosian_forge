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
@overload(operator.setitem)
def literal_list_setitem(lst, index, value):
    if isinstance(lst, types.LiteralList):
        raise errors.TypingError('Cannot mutate a literal list')