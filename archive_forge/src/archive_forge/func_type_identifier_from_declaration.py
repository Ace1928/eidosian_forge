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
def type_identifier_from_declaration(decl, scope=None):
    key = (decl, scope)
    safe = _type_identifier_cache.get(key)
    if safe is None:
        safe = decl
        if scope:
            safe = scope.mangle(prefix='', name=safe)
        safe = re.sub(' +', ' ', safe)
        safe = re.sub(' ?([^a-zA-Z0-9_]) ?', '\\1', safe)
        safe = _escape_special_type_characters(safe)
        safe = cap_length(re.sub('[^a-zA-Z0-9_]', lambda x: '__%X' % ord(x.group(0)), safe))
        _type_identifier_cache[key] = safe
    return safe