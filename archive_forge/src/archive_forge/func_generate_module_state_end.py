from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def generate_module_state_end(self, env, modules, globalstate):
    module_state = globalstate['module_state']
    module_state_defines = globalstate['module_state_defines']
    module_state_clear = globalstate['module_state_clear']
    module_state_traverse = globalstate['module_state_traverse']
    module_state.putln('} %s;' % Naming.modulestate_cname)
    module_state.putln('')
    module_state.putln('#if CYTHON_USE_MODULE_STATE')
    module_state.putln('#ifdef __cplusplus')
    module_state.putln('namespace {')
    module_state.putln('extern struct PyModuleDef %s;' % Naming.pymoduledef_cname)
    module_state.putln('} /* anonymous namespace */')
    module_state.putln('#else')
    module_state.putln('static struct PyModuleDef %s;' % Naming.pymoduledef_cname)
    module_state.putln('#endif')
    module_state.putln('')
    module_state.putln('#define %s(o) ((%s *)__Pyx_PyModule_GetState(o))' % (Naming.modulestate_cname, Naming.modulestate_cname))
    module_state.putln('')
    module_state.putln('#define %s (%s(PyState_FindModule(&%s)))' % (Naming.modulestateglobal_cname, Naming.modulestate_cname, Naming.pymoduledef_cname))
    module_state.putln('')
    module_state.putln('#define %s (PyState_FindModule(&%s))' % (env.module_cname, Naming.pymoduledef_cname))
    module_state.putln('#else')
    module_state.putln('static %s %s_static =' % (Naming.modulestate_cname, Naming.modulestateglobal_cname))
    module_state.putln('#ifdef __cplusplus')
    module_state.putln('    {};')
    module_state.putln('#else')
    module_state.putln('    {0};')
    module_state.putln('#endif')
    module_state.putln('static %s *%s = &%s_static;' % (Naming.modulestate_cname, Naming.modulestateglobal_cname, Naming.modulestateglobal_cname))
    module_state.putln('#endif')
    module_state_clear.putln('return 0;')
    module_state_clear.putln('}')
    module_state_clear.putln('#endif')
    module_state_traverse.putln('return 0;')
    module_state_traverse.putln('}')
    module_state_traverse.putln('#endif')