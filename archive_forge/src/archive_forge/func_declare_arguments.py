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
def declare_arguments(self, env):
    for arg in self.args:
        if not arg.name:
            error(arg.pos, 'Missing argument name')
        if arg.needs_conversion:
            arg.entry = env.declare_var(arg.name, arg.type, arg.pos)
            if arg.type.is_pyobject:
                arg.entry.init = '0'
        else:
            arg.entry = self.declare_argument(env, arg)
        arg.entry.is_arg = 1
        arg.entry.used = 1
        arg.entry.is_self_arg = arg.is_self_arg
    self.declare_python_arg(env, self.star_arg)
    self.declare_python_arg(env, self.starstar_arg)