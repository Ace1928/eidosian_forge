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
def declared_type(ctype):
    type_displayname = str(ctype.declaration_code('', for_display=True))
    if ctype.is_pyobject:
        arg_ctype = type_name = type_displayname
        if ctype.is_builtin_type:
            arg_ctype = ctype.name
        elif not ctype.is_extension_type:
            type_name = 'object'
            type_displayname = None
        else:
            type_displayname = repr(type_displayname)
    elif ctype is c_bint_type:
        type_name = arg_ctype = 'bint'
    else:
        type_name = arg_ctype = type_displayname
        if ctype is c_double_type:
            type_displayname = 'float'
        else:
            type_displayname = repr(type_displayname)
    return (type_name, arg_ctype, type_displayname)