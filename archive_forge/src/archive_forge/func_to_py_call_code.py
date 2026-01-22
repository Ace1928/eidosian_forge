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
def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
    func = self.to_py_function if to_py_function is None else to_py_function
    if self.is_string or self.is_pyunicode_ptr:
        return '%s = %s(%s)' % (result_code, func, source_code)
    target_is_tuple = result_type.is_builtin_type and result_type.name == 'tuple'
    return '%s = %s(%s, %s)' % (result_code, self.to_tuple_function if target_is_tuple else func, source_code, self.size)