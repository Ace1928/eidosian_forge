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
def initialize_privates_to_nan(self, code, exclude=None):
    first = True
    for entry, (op, lastprivate) in sorted(self.privates.items()):
        if not op and (not exclude or entry != exclude):
            invalid_value = entry.type.invalid_value()
            if invalid_value:
                if first:
                    code.putln('/* Initialize private variables to invalid values */')
                    first = False
                code.putln('%s = %s;' % (entry.cname, entry.type.cast_code(invalid_value)))