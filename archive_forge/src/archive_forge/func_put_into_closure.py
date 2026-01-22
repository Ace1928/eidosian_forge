from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def put_into_closure(entry):
    if entry.in_closure:
        if entry.type.is_array:
            assert entry.type.size is not None
            code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStringH', 'StringTools.c'))
            code.putln('memcpy({0}, {1}, sizeof({0}));'.format(entry.cname, entry.original_cname))
        else:
            code.putln('%s = %s;' % (entry.cname, entry.original_cname))
        if entry.type.is_memoryviewslice:
            entry.type.generate_incref_memoryviewslice(code, entry.cname, True)
        elif entry.xdecref_cleanup:
            code.put_var_xincref(entry)
            code.put_var_xgiveref(entry)
        else:
            code.put_var_incref(entry)
            code.put_var_giveref(entry)