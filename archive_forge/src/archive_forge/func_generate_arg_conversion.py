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
def generate_arg_conversion(self, arg, code):
    old_type = arg.hdr_type
    new_type = arg.type
    if old_type.is_pyobject:
        if arg.default:
            code.putln('if (%s) {' % arg.hdr_cname)
        else:
            code.putln('assert(%s); {' % arg.hdr_cname)
        self.generate_arg_conversion_from_pyobject(arg, code)
        code.putln('}')
    elif new_type.is_pyobject:
        self.generate_arg_conversion_to_pyobject(arg, code)
    elif new_type.assignable_from(old_type):
        code.putln('%s = %s;' % (arg.entry.cname, arg.hdr_cname))
    else:
        error(arg.pos, "Cannot convert 1 argument from '%s' to '%s'" % (old_type, new_type))