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
def generate_arg_type_test(self, arg, code):
    if arg.type.typeobj_is_available():
        code.globalstate.use_utility_code(UtilityCode.load_cached('ArgTypeTest', 'FunctionArguments.c'))
        typeptr_cname = arg.type.typeptr_cname
        arg_code = '((PyObject *)%s)' % arg.entry.cname
        code.putln('if (unlikely(!__Pyx_ArgTypeTest(%s, %s, %d, %s, %s))) %s' % (arg_code, typeptr_cname, arg.accept_none, arg.name_cstring, arg.type.is_builtin_type and arg.type.require_exact, code.error_goto(arg.pos)))
    else:
        error(arg.pos, 'Cannot test type of extern C class without type object name specification')