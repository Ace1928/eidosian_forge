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
def get_getbuffer_call(code, obj_cname, buffer_aux, buffer_type):
    ndim = buffer_type.ndim
    cast = int(buffer_type.cast)
    flags = get_flags(buffer_aux, buffer_type)
    pybuffernd_struct = buffer_aux.buflocal_nd_var.cname
    dtype_typeinfo = get_type_information_cname(code, buffer_type.dtype)
    code.globalstate.use_utility_code(acquire_utility_code)
    return '__Pyx_GetBufferAndValidate(&%(pybuffernd_struct)s.rcbuffer->pybuffer, (PyObject*)%(obj_cname)s, &%(dtype_typeinfo)s, %(flags)s, %(ndim)d, %(cast)d, __pyx_stack)' % locals()