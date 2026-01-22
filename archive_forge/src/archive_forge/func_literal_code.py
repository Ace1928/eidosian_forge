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
def literal_code(self, value):
    if self.is_string:
        assert isinstance(value, str)
        return '"%s"' % StringEncoding.escape_byte_string(value)
    return str(value)