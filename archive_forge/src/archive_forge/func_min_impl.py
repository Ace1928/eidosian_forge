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
def min_impl(indval1, indval2):
    if np.isnan(indval1.value):
        if np.isnan(indval2.value):
            if indval1.index < indval2.index:
                return indval1
            else:
                return indval2
        else:
            return indval1
    elif np.isnan(indval2.value):
        return indval2
    elif indval1.value > indval2.value:
        return indval2
    elif indval1.value == indval2.value:
        if indval1.index < indval2.index:
            return indval1
        else:
            return indval2
    return indval1