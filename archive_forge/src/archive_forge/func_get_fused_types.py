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
def get_fused_types(self, result=None, seen=None, include_function_return_type=False):
    if result is None:
        result = []
        seen = set()
    if self.namespace:
        self.namespace.get_fused_types(result, seen)
    if self.templates:
        for T in self.templates:
            T.get_fused_types(result, seen)
    return result