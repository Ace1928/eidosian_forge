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
def dtype_object_conversion_funcs(self, env):
    get_function = '__pyx_memview_get_%s' % self.dtype_name
    set_function = '__pyx_memview_set_%s' % self.dtype_name
    context = dict(get_function=get_function, set_function=set_function)
    if self.dtype.is_pyobject:
        utility_name = 'MemviewObjectToObject'
    else:
        self.dtype.create_to_py_utility_code(env)
        to_py_function = self.dtype.to_py_function
        from_py_function = None
        if not self.dtype.is_const:
            self.dtype.create_from_py_utility_code(env)
            from_py_function = self.dtype.from_py_function
        if not (to_py_function or from_py_function):
            return ('NULL', 'NULL')
        if not to_py_function:
            get_function = 'NULL'
        if not from_py_function:
            set_function = 'NULL'
        utility_name = 'MemviewDtypeToObject'
        error_condition = self.dtype.error_condition('value') or 'PyErr_Occurred()'
        context.update(to_py_function=to_py_function, from_py_function=from_py_function, dtype=self.dtype.empty_declaration_code(), error_condition=error_condition)
    utility = TempitaUtilityCode.load_cached(utility_name, 'MemoryView_C.c', context=context)
    env.use_utility_code(utility)
    return (get_function, set_function)