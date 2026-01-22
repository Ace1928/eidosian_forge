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
def find_buffer_types(scope):
    if scope in visited_scopes:
        return
    visited_scopes.add(scope)
    for m in scope.cimported_modules:
        find_buffer_types(m)
    for e in scope.type_entries:
        if isinstance(e.utility_code_definition, CythonUtilityCode):
            continue
        t = e.type
        if t.is_extension_type:
            if scope is cython_scope and (not e.used):
                continue
            release = get = None
            for x in t.scope.pyfunc_entries:
                if x.name == u'__getbuffer__':
                    get = x.func_cname
                elif x.name == u'__releasebuffer__':
                    release = x.func_cname
            if get:
                types.append((t.typeptr_cname, get, release))