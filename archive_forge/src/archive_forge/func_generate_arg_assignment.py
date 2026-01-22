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
def generate_arg_assignment(self, arg, item, code):
    if arg.type.is_pyobject:
        if arg.is_generic:
            item = PyrexTypes.typecast(arg.type, PyrexTypes.py_object_type, item)
        entry = arg.entry
        code.putln('%s = %s;' % (entry.cname, item))
    elif arg.type.from_py_function:
        if arg.default:
            code.putln('if (%s) {' % item)
        code.putln(arg.type.from_py_call_code(item, arg.entry.cname, arg.pos, code))
        if arg.default:
            code.putln('} else {')
            code.putln('%s = %s;' % (arg.entry.cname, arg.calculate_default_value_code(code)))
            if arg.type.is_memoryviewslice:
                code.put_var_incref_memoryviewslice(arg.entry, have_gil=True)
            code.putln('}')
    else:
        error(arg.pos, "Cannot convert Python object argument to type '%s'" % arg.type)