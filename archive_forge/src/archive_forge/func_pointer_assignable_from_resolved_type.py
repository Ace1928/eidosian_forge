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
def pointer_assignable_from_resolved_type(self, rhs_type):
    if rhs_type is error_type:
        return 1
    if not rhs_type.is_cfunction:
        return 0
    return rhs_type.same_c_signature_as_resolved_type(self, exact_semantics=False) and (not (self.nogil and (not rhs_type.nogil)))