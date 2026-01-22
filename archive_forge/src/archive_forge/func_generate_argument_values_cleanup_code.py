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
def generate_argument_values_cleanup_code(self, code):
    if not self.needs_values_cleanup:
        return
    loop_var = Naming.quick_temp_cname
    code.putln('{')
    code.putln('Py_ssize_t %s;' % loop_var)
    code.putln('for (%s=0; %s < (Py_ssize_t)(sizeof(values)/sizeof(values[0])); ++%s) {' % (loop_var, loop_var, loop_var))
    code.putln('__Pyx_Arg_XDECREF_%s(values[%s]);' % (self.signature.fastvar, loop_var))
    code.putln('}')
    code.putln('}')