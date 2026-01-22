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
def subtype_of_resolved_type(self, other_type):
    if other_type.is_extension_type or other_type.is_builtin_type:
        return self is other_type or (self.base_type and self.base_type.subtype_of(other_type))
    else:
        return other_type is py_object_type