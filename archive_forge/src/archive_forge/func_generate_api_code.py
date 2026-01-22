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
def generate_api_code(self, env, options, result):

    def api_entries(entries, pxd=0):
        return [entry for entry in entries if entry.api or (pxd and entry.defined_in_pxd)]
    api_vars = api_entries(env.var_entries)
    api_funcs = api_entries(env.cfunc_entries)
    api_extension_types = api_entries(env.c_class_entries)
    if api_vars or api_funcs or api_extension_types:
        result.api_file = replace_suffix_encoded(result.c_file, '_api.h')
        self.assure_safe_target(result.api_file)
        h_code = Code.CCodeWriter()
        c_code_config = generate_c_code_config(env, options)
        Code.GlobalState(h_code, self, c_code_config)
        h_code.put_generated_by()
        api_guard = self.api_name(Naming.api_guard_prefix, env)
        h_code.put_h_guard(api_guard)
        h_code.putln('#ifdef __MINGW64__')
        h_code.putln('#define MS_WIN64')
        h_code.putln('#endif')
        h_code.putln('#include "Python.h"')
        if result.h_file:
            h_filename = os.path.basename(result.h_file)
            h_filename = as_encoded_filename(h_filename)
            h_code.putln('#include %s' % h_filename.as_c_string_literal())
        if api_extension_types:
            h_code.putln('')
            for entry in api_extension_types:
                type = entry.type
                h_code.putln('static PyTypeObject *%s = 0;' % type.typeptr_cname)
                h_code.putln('#define %s (*%s)' % (type.typeobj_cname, type.typeptr_cname))
        if api_funcs:
            h_code.putln('')
            for entry in api_funcs:
                type = CPtrType(entry.type)
                cname = env.mangle(Naming.func_prefix_api, entry.name)
                h_code.putln('static %s = 0;' % type.declaration_code(cname))
                h_code.putln('#define %s %s' % (entry.name, cname))
        if api_vars:
            h_code.putln('')
            for entry in api_vars:
                type = CPtrType(entry.type)
                cname = env.mangle(Naming.varptr_prefix_api, entry.name)
                h_code.putln('static %s = 0;' % type.declaration_code(cname))
                h_code.putln('#define %s (*%s)' % (entry.name, cname))
        if api_vars:
            h_code.put(UtilityCode.load_as_string('VoidPtrImport', 'ImportExport.c')[1])
        if api_funcs:
            h_code.put(UtilityCode.load_as_string('FunctionImport', 'ImportExport.c')[1])
        if api_extension_types:
            h_code.put(UtilityCode.load_as_string('TypeImport', 'ImportExport.c')[0])
            h_code.put(UtilityCode.load_as_string('TypeImport', 'ImportExport.c')[1])
        h_code.putln('')
        h_code.putln('static int %s(void) {' % self.api_name('import', env))
        h_code.putln('PyObject *module = 0;')
        h_code.putln('module = PyImport_ImportModule(%s);' % env.qualified_name.as_c_string_literal())
        h_code.putln('if (!module) goto bad;')
        for entry in api_funcs:
            cname = env.mangle(Naming.func_prefix_api, entry.name)
            sig = entry.type.signature_string()
            h_code.putln('if (__Pyx_ImportFunction_%s(module, %s, (void (**)(void))&%s, "%s") < 0) goto bad;' % (Naming.cyversion, entry.name.as_c_string_literal(), cname, sig))
        for entry in api_vars:
            cname = env.mangle(Naming.varptr_prefix_api, entry.name)
            sig = entry.type.empty_declaration_code()
            h_code.putln('if (__Pyx_ImportVoidPtr_%s(module, %s, (void **)&%s, "%s") < 0) goto bad;' % (Naming.cyversion, entry.name.as_c_string_literal(), cname, sig))
        with ModuleImportGenerator(h_code, imported_modules={env.qualified_name: 'module'}) as import_generator:
            for entry in api_extension_types:
                self.generate_type_import_call(entry.type, h_code, import_generator, error_code='goto bad;')
        h_code.putln('Py_DECREF(module); module = 0;')
        h_code.putln('return 0;')
        h_code.putln('bad:')
        h_code.putln('Py_XDECREF(module);')
        h_code.putln('return -1;')
        h_code.putln('}')
        h_code.putln('')
        h_code.putln('#endif /* !%s */' % api_guard)
        f = open_new_file(result.api_file)
        try:
            h_code.copyto(f)
        finally:
            f.close()