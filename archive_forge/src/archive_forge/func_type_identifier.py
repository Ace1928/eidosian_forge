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
def type_identifier(type, pyrex=False):
    scope = None
    decl = type.empty_declaration_code(pyrex=pyrex)
    entry = getattr(type, 'entry', None)
    if entry and entry.scope:
        scope = entry.scope
    return type_identifier_from_declaration(decl, scope=scope)