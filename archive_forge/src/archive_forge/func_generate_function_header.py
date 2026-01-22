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
def generate_function_header(self, code, proto=False):
    header = 'static PyObject *%s(__pyx_CoroutineObject *%s, CYTHON_UNUSED PyThreadState *%s, PyObject *%s)' % (self.entry.func_cname, Naming.generator_cname, Naming.local_tstate_cname, Naming.sent_value_cname)
    if proto:
        code.putln('%s; /* proto */' % header)
    else:
        code.putln('%s /* generator body */\n{' % header)