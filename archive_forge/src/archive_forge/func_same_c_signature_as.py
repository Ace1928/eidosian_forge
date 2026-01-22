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
def same_c_signature_as(self, other_type, as_cmethod=0):
    return self.same_c_signature_as_resolved_type(other_type.resolve(), as_cmethod)