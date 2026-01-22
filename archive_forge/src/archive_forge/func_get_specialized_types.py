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
def get_specialized_types(type):
    """
    Return a list of specialized types in their declared order.
    """
    assert type.is_fused
    if isinstance(type, FusedType):
        result = list(type.types)
        for specialized_type in result:
            specialized_type.specialization_string = specialized_type.typeof_name()
    else:
        result = []
        for cname, f2s in get_all_specialized_permutations(type.get_fused_types()):
            specialized_type = type.specialize(f2s)
            specialized_type.specialization_string = specialization_signature_string(type, f2s)
            result.append(specialized_type)
    return result