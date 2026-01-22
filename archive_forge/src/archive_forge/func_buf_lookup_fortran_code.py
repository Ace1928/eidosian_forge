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
def buf_lookup_fortran_code(proto, defin, name, nd):
    """
    Like C lookup, but the first index is optimized instead.
    """
    if nd == 1:
        proto.putln('#define %s(type, buf, i0, s0) ((type)buf + i0)' % name)
    else:
        args = ', '.join(['i%d, s%d' % (i, i) for i in range(nd)])
        offset = ' + '.join(['i%d * s%d' % (i, i) for i in range(1, nd)])
        proto.putln('#define %s(type, buf, %s) ((type)((char*)buf + %s) + i%d)' % (name, args, offset, 0))