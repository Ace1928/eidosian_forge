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
def generate_arg_none_check(self, arg, code):
    if arg.type.is_memoryviewslice:
        cname = '%s.memview' % arg.entry.cname
    else:
        cname = arg.entry.cname
    code.putln('if (unlikely(((PyObject *)%s) == Py_None)) {' % cname)
    code.putln('PyErr_Format(PyExc_TypeError, "Argument \'%%.%ds\' must not be None", %s); %s' % (max(200, len(arg.name_cstring)), arg.name_cstring, code.error_goto(arg.pos)))
    code.putln('}')