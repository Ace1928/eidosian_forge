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
def generate_decref_clear(self, code, cname, clear_before_decref, nanny, have_gil):
    self._generate_decref(code, cname, nanny, null_check=False, clear=True, clear_before_decref=clear_before_decref)