import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
@overload(dict_vs_cache_vs_parametrized)
def ol_dict_vs_cache_vs_parametrized(v):
    typ = v

    def objmode_vs_cache_vs_parametrized_impl(v):
        d = typed.Dict.empty(types.unicode_type, typ)
        d['data'] = v
    return objmode_vs_cache_vs_parametrized_impl