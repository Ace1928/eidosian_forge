from collections import namedtuple
import math
from functools import reduce
import numpy as np
import operator
import warnings
from llvmlite import ir
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, cgutils
from numba.core.extending import overload, intrinsic
from numba.core.typeconv import Conversion
from numba.core.errors import (TypingError, LoweringError,
from numba.misc.special import literal_unroll
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.typing.builtins import IndexValue, IndexValueType
from numba.extending import overload, register_jitable
@overload(filter)
def ol_filter(func, iterable):
    if func is None or isinstance(func, types.NoneType):

        def impl(func, iterable):
            for x in iterable:
                if x:
                    yield x
    else:

        def impl(func, iterable):
            for x in iterable:
                if func(x):
                    yield x
    return impl