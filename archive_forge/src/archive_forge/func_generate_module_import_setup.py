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
def generate_module_import_setup(self, env, code):
    module_path = env.directives['set_initial_path']
    if module_path == 'SOURCEFILE':
        module_path = self.pos[0].filename
    if module_path:
        code.putln('if (!CYTHON_PEP489_MULTI_PHASE_INIT) {')
        code.putln('if (PyObject_SetAttrString(%s, "__file__", %s) < 0) %s;' % (env.module_cname, code.globalstate.get_py_string_const(EncodedString(decode_filename(module_path))).cname, code.error_goto(self.pos)))
        code.putln('}')
        if env.is_package:
            code.putln('if (!CYTHON_PEP489_MULTI_PHASE_INIT) {')
            temp = code.funcstate.allocate_temp(py_object_type, True)
            code.putln('%s = Py_BuildValue("[O]", %s); %s' % (temp, code.globalstate.get_py_string_const(EncodedString(decode_filename(os.path.dirname(module_path)))).cname, code.error_goto_if_null(temp, self.pos)))
            code.put_gotref(temp, py_object_type)
            code.putln('if (PyObject_SetAttrString(%s, "__path__", %s) < 0) %s;' % (env.module_cname, temp, code.error_goto(self.pos)))
            code.put_decref_clear(temp, py_object_type)
            code.funcstate.release_temp(temp)
            code.putln('}')
    elif env.is_package:
        code.putln('if (!CYTHON_PEP489_MULTI_PHASE_INIT) {')
        code.globalstate.use_utility_code(UtilityCode.load('SetPackagePathFromImportLib', 'ImportExport.c'))
        code.putln(code.error_goto_if_neg('__Pyx_SetPackagePathFromImportLib(%s)' % code.globalstate.get_py_string_const(EncodedString(self.full_module_name)).cname, self.pos))
        code.putln('}')
    fq_module_name = self.full_module_name
    if fq_module_name.endswith('.__init__'):
        fq_module_name = EncodedString(fq_module_name[:-len('.__init__')])
    fq_module_name_cstring = fq_module_name.as_c_string_literal()
    code.putln('#if PY_MAJOR_VERSION >= 3')
    code.putln('{')
    code.putln('PyObject *modules = PyImport_GetModuleDict(); %s' % code.error_goto_if_null('modules', self.pos))
    code.putln('if (!PyDict_GetItemString(modules, %s)) {' % fq_module_name_cstring)
    code.putln(code.error_goto_if_neg('PyDict_SetItemString(modules, %s, %s)' % (fq_module_name_cstring, env.module_cname), self.pos))
    code.putln('}')
    code.putln('}')
    code.putln('#endif')