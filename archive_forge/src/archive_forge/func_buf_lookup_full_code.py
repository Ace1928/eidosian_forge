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
def buf_lookup_full_code(proto, defin, name, nd):
    """
    Generates a buffer lookup function for the right number
    of dimensions. The function gives back a void* at the right location.
    """
    macroargs = ', '.join(['i%d, s%d, o%d' % (i, i, i) for i in range(nd)])
    proto.putln('#define %s(type, buf, %s) (type)(%s_imp(buf, %s))' % (name, macroargs, name, macroargs))
    funcargs = ', '.join(['Py_ssize_t i%d, Py_ssize_t s%d, Py_ssize_t o%d' % (i, i, i) for i in range(nd)])
    proto.putln('static CYTHON_INLINE void* %s_imp(void* buf, %s);' % (name, funcargs))
    defin.putln(dedent('\n        static CYTHON_INLINE void* %s_imp(void* buf, %s) {\n          char* ptr = (char*)buf;\n        ') % (name, funcargs) + ''.join([dedent('          ptr += s%d * i%d;\n          if (o%d >= 0) ptr = *((char**)ptr) + o%d;\n        ') % (i, i, i, i) for i in range(nd)]) + '\nreturn ptr;\n}')