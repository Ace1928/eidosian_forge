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
def independent_spanning_type(type1, type2):
    if type1.is_reference ^ type2.is_reference:
        if type1.is_reference:
            type1 = type1.ref_base_type
        else:
            type2 = type2.ref_base_type
    resolved_type1 = type1.resolve()
    resolved_type2 = type2.resolve()
    if resolved_type1 == resolved_type2:
        return type1
    elif (resolved_type1 is c_bint_type or resolved_type2 is c_bint_type) and (type1.is_numeric and type2.is_numeric):
        return py_object_type
    span_type = _spanning_type(type1, type2)
    if span_type is None:
        return error_type
    return span_type