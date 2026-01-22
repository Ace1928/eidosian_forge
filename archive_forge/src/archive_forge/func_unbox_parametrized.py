import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
@unbox(ParametrizedType)
def unbox_parametrized(typ, obj, context):
    return context.unbox(types.UniTuple(typ.dtype, len(typ)), obj)