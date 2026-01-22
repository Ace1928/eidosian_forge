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
def put_init_vars(entry, code):
    bufaux = entry.buffer_aux
    pybuffernd_struct = bufaux.buflocal_nd_var.cname
    pybuffer_struct = bufaux.rcbuf_var.cname
    code.putln('%s.pybuffer.buf = NULL;' % pybuffer_struct)
    code.putln('%s.refcount = 0;' % pybuffer_struct)
    code.putln('%s.data = NULL;' % pybuffernd_struct)
    code.putln('%s.rcbuffer = &%s;' % (pybuffernd_struct, pybuffer_struct))