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
def specialize_here(self, pos, env, template_values=None):
    if len(template_values) != 1:
        error(pos, "'%s' takes exactly one template argument." % self.name)
        return error_type
    if template_values[0] is None:
        return None
    return template_values[0].resolve()