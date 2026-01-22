from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
def decvar(type, prefix):
    cname = scope.mangle(prefix, name)
    aux_var = scope.declare_var(name=None, cname=cname, type=type, pos=node.pos)
    if entry.is_arg:
        aux_var.used = True
    return aux_var