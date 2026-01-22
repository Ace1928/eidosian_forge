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
def axes_to_code(self):
    """Return a list of code constants for each axis"""
    from . import MemoryView
    d = MemoryView._spec_to_const
    return ['(%s | %s)' % (d[a], d[p]) for a, p in self.axes]