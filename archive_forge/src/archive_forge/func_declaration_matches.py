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
def declaration_matches(self, entry, kind):
    if not entry.is_type:
        return 0
    type = entry.type
    if kind == 'class':
        if not type.is_extension_type:
            return 0
    else:
        if not type.is_struct_or_union:
            return 0
        if kind != type.kind:
            return 0
    return 1