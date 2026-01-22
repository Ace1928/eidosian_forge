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
def c_ptr_type(base_type):
    if base_type.is_reference:
        base_type = base_type.ref_base_type
    return _construct_type_from_base(CPtrType, base_type)