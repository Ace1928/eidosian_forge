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
def base_declaration_code(self, base_code, entity_code):
    if entity_code:
        return '%s %s' % (base_code, entity_code)
    else:
        return base_code