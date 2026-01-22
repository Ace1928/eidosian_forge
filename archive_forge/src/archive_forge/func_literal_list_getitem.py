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
@overload(operator.getitem)
def literal_list_getitem(lst, *args):
    if not isinstance(lst, types.LiteralList):
        return
    msg = 'Cannot __getitem__ on a literal list, return type cannot be statically determined.'
    raise errors.TypingError(msg)