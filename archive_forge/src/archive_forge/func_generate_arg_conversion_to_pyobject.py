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
def generate_arg_conversion_to_pyobject(self, arg, code):
    old_type = arg.hdr_type
    func = old_type.to_py_function
    if func:
        code.putln('%s = %s(%s); %s' % (arg.entry.cname, func, arg.hdr_cname, code.error_goto_if_null(arg.entry.cname, arg.pos)))
        code.put_var_gotref(arg.entry)
    else:
        error(arg.pos, "Cannot convert argument of type '%s' to Python object" % old_type)