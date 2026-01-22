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
def with_with_gil(self, with_gil):
    if with_gil == self.with_gil:
        return self
    else:
        return CFuncType(self.return_type, self.args, self.has_varargs, self.exception_value, self.exception_check, self.calling_convention, self.nogil, with_gil, self.is_overridable, self.optional_arg_count, self.is_const_method, self.is_static_method, self.templates, self.is_strict_signature)