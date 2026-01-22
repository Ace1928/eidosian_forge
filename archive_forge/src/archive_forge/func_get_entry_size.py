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
def get_entry_size(context, set_type):
    """
    Return the entry size for the given set type.
    """
    llty = context.get_data_type(types.SetEntry(set_type))
    return context.get_abi_sizeof(llty)