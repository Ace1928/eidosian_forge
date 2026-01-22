import sys
import operator
import numpy as np
from llvmlite.ir import IntType, Constant
from numba.core.cgutils import is_nonelike
from numba.core.extending import (
from numba.core.imputils import (lower_constant, lower_cast, lower_builtin,
from numba.core.datamodel import register_default, StructModel
from numba.core import types, cgutils
from numba.core.utils import PYVERSION
from numba.core.pythonapi import (
from numba._helperlib import c_helpers
from numba.cpython.hashing import _Py_hash_t
from numba.core.unsafe.bytes import memcpy_region
from numba.core.errors import TypingError
from numba.cpython.unicode_support import (_Py_TOUPPER, _Py_TOLOWER, _Py_UCS4,
from numba.cpython import slicing
@box(types.UnicodeType)
def box_unicode_str(typ, val, c):
    """
    Convert a native unicode structure to a unicode string
    """
    uni_str = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    res = c.pyapi.string_from_kind_and_data(uni_str.kind, uni_str.data, uni_str.length)
    c.pyapi.object_hash(res)
    c.context.nrt.decref(c.builder, typ, val)
    return res