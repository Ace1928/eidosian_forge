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
def cap_length(s, max_len=63):
    if len(s) <= max_len:
        return s
    hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
    return '%s__%s__etc' % (hash_prefix, s[:max_len - 17])