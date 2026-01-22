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
def check_nullary_constructor(self, pos, msg='stack allocated'):
    constructor = self.scope.lookup(u'<init>')
    if constructor is not None and best_match([], constructor.all_alternatives()) is None:
        error(pos, 'C++ class must have a nullary constructor to be %s' % msg)