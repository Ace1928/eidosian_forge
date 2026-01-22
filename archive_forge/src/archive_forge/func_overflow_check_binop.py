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
def overflow_check_binop(self, binop, env, const_rhs=False):
    env.use_utility_code(UtilityCode.load('Common', 'Overflow.c'))
    type = self.empty_declaration_code()
    name = self.specialization_name()
    if binop == 'lshift':
        env.use_utility_code(TempitaUtilityCode.load_cached('LeftShift', 'Overflow.c', context={'TYPE': type, 'NAME': name, 'SIGNED': self.signed}))
    else:
        if const_rhs:
            binop += '_const'
        if type in ('int', 'long', 'long long'):
            env.use_utility_code(TempitaUtilityCode.load_cached('BaseCaseSigned', 'Overflow.c', context={'INT': type, 'NAME': name}))
        elif type in ('unsigned int', 'unsigned long', 'unsigned long long'):
            env.use_utility_code(TempitaUtilityCode.load_cached('BaseCaseUnsigned', 'Overflow.c', context={'UINT': type, 'NAME': name}))
        elif self.rank <= 1:
            return '__Pyx_%s_%s_no_overflow' % (binop, name)
        else:
            _load_overflow_base(env)
            env.use_utility_code(TempitaUtilityCode.load_cached('SizeCheck', 'Overflow.c', context={'TYPE': type, 'NAME': name}))
            env.use_utility_code(TempitaUtilityCode.load_cached('Binop', 'Overflow.c', context={'TYPE': type, 'NAME': name, 'BINOP': binop}))
    return '__Pyx_%s_%s_checking_overflow' % (binop, name)