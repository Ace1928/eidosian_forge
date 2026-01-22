from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def is_promotion(src_type, dst_type):
    if src_type.is_numeric:
        if dst_type.same_as(c_int_type):
            unsigned = not src_type.signed
            return src_type.is_enum or (src_type.is_int and unsigned + src_type.rank < dst_type.rank)
        elif dst_type.same_as(c_double_type):
            return src_type.is_float and src_type.rank <= dst_type.rank
    return False