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
def getbuffer_init(self, code):
    py_buffer, obj_type = self._get_py_buffer_info()
    view = py_buffer.cname
    if obj_type and obj_type.is_pyobject:
        code.put_init_to_py_none('%s->obj' % view, obj_type)
        code.put_giveref('%s->obj' % view, obj_type)
    else:
        code.putln('%s->obj = NULL;' % view)