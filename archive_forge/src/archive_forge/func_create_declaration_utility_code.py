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
def create_declaration_utility_code(self, env):
    if self.real_type.is_float:
        env.use_utility_code(UtilityCode.load_cached('Header', 'Complex.c'))
    utility_code_context = self._utility_code_context()
    env.use_utility_code(UtilityCode.load_cached('RealImag' + self.implementation_suffix, 'Complex.c'))
    env.use_utility_code(TempitaUtilityCode.load_cached('Declarations', 'Complex.c', utility_code_context))
    env.use_utility_code(TempitaUtilityCode.load_cached('Arithmetic', 'Complex.c', utility_code_context))
    return True