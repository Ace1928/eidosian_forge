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
def cv_string(self):
    cvstring = ''
    if self.is_const:
        cvstring = 'const ' + cvstring
    if self.is_volatile:
        cvstring = 'volatile ' + cvstring
    return cvstring