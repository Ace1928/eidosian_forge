from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def type_may_always_need_nrt(ty):
    if not isinstance(ty, types.Array):
        dmm = self.context.data_model_manager
        if dmm[ty].contains_nrt_meminfo():
            return True
    return False