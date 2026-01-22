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
def generate_incref_memoryviewslice(self, code, slice_cname, have_gil):
    code.putln('__PYX_INC_MEMVIEW(&%s, %d);' % (slice_cname, int(have_gil)))