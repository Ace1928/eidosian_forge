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
def assignable_from_resolved_type(self, other_type):
    if other_type is error_type:
        return True
    elif other_type.is_cpp_class:
        return other_type.is_subclass(self)
    elif other_type.is_string and self.cname in cpp_string_conversions:
        return True