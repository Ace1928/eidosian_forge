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
@lower_builtin('iternext', types.SetIter)
@iternext_impl(RefType.BORROWED)
def iternext_listiter(context, builder, sig, args, result):
    inst = SetIterInstance(context, builder, sig.args[0], args[0])
    inst.iternext(result)