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
def get_to_py_type_conversion(self):
    if self.rank < list(rank_to_type_name).index('int'):
        return 'PyInt_FromLong'
    else:
        Prefix = 'Int'
        SignWord = ''
        TypeName = 'Long'
        if not self.signed:
            Prefix = 'Long'
            SignWord = 'Unsigned'
        if self.rank >= list(rank_to_type_name).index('PY_LONG_LONG'):
            Prefix = 'Long'
            TypeName = 'LongLong'
        return 'Py%s_From%s%s' % (Prefix, SignWord, TypeName)