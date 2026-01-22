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
def assure_gil(code_path, code=code):
    if not gil_owned[code_path]:
        if not gil_owned['gil_state_declared']:
            gilstate_decl.declare_gilstate()
            gil_owned['gil_state_declared'] = True
        code.put_ensure_gil(declare_gilstate=False)
        gil_owned[code_path] = True