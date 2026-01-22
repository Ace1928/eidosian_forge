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
def generate_h_code(self, env, options, result):

    def h_entries(entries, api=0, pxd=0):
        return [entry for entry in entries if entry.visibility == 'public' or (api and entry.api) or (pxd and entry.defined_in_pxd)]
    h_types = h_entries(env.type_entries, api=1)
    h_vars = h_entries(env.var_entries)
    h_funcs = h_entries(env.cfunc_entries)
    h_extension_types = h_entries(env.c_class_entries)
    if h_types or h_vars or h_funcs or h_extension_types:
        result.h_file = replace_suffix_encoded(result.c_file, '.h')
        self.assure_safe_target(result.h_file)
        h_code_writer = Code.CCodeWriter()
        c_code_config = generate_c_code_config(env, options)
        globalstate = Code.GlobalState(h_code_writer, self, c_code_config)
        globalstate.initialize_main_h_code()
        h_code_start = globalstate.parts['h_code']
        h_code_main = globalstate.parts['type_declarations']
        h_code_end = globalstate.parts['end']
        if options.generate_pxi:
            result.i_file = replace_suffix_encoded(result.c_file, '.pxi')
            i_code = Code.PyrexCodeWriter(result.i_file)
        else:
            i_code = None
        h_code_start.put_generated_by()
        h_guard = self.api_name(Naming.h_guard_prefix, env)
        h_code_start.put_h_guard(h_guard)
        h_code_start.putln('')
        h_code_start.putln('#include "Python.h"')
        self.generate_type_header_code(h_types, h_code_start)
        if options.capi_reexport_cincludes:
            self.generate_includes(env, [], h_code_start)
        h_code_start.putln('')
        api_guard = self.api_name(Naming.api_guard_prefix, env)
        h_code_start.putln('#ifndef %s' % api_guard)
        h_code_start.putln('')
        self.generate_extern_c_macro_definition(h_code_start, env.is_cpp())
        h_code_start.putln('')
        self.generate_dl_import_macro(h_code_start)
        if h_extension_types:
            h_code_main.putln('')
            for entry in h_extension_types:
                self.generate_cclass_header_code(entry.type, h_code_main)
                if i_code:
                    self.generate_cclass_include_code(entry.type, i_code)
        if h_funcs:
            h_code_main.putln('')
            for entry in h_funcs:
                self.generate_public_declaration(entry, h_code_main, i_code)
        if h_vars:
            h_code_main.putln('')
            for entry in h_vars:
                self.generate_public_declaration(entry, h_code_main, i_code)
        h_code_main.putln('')
        h_code_main.putln('#endif /* !%s */' % api_guard)
        h_code_main.putln('')
        h_code_main.putln('/* WARNING: the interface of the module init function changed in CPython 3.5. */')
        h_code_main.putln('/* It now returns a PyModuleDef instance instead of a PyModule instance. */')
        h_code_main.putln('')
        h_code_main.putln('#if PY_MAJOR_VERSION < 3')
        if env.module_name.isascii():
            py2_mod_name = env.module_name
        else:
            py2_mod_name = env.module_name.encode('ascii', errors='ignore').decode('utf-8')
            h_code_main.putln('#error "Unicode module names are not supported in Python 2";')
        h_code_main.putln('PyMODINIT_FUNC init%s(void);' % py2_mod_name)
        h_code_main.putln('#else')
        py3_mod_func_name = self.mod_init_func_cname('PyInit', env)
        warning_string = EncodedString('Use PyImport_AppendInittab("%s", %s) instead of calling %s directly.' % (py2_mod_name, py3_mod_func_name, py3_mod_func_name))
        h_code_main.putln('/* WARNING: %s from Python 3.5 */' % warning_string.rstrip('.'))
        h_code_main.putln('PyMODINIT_FUNC %s(void);' % py3_mod_func_name)
        h_code_main.putln('')
        h_code_main.putln('#if PY_VERSION_HEX >= 0x03050000 && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || (defined(__cplusplus) && __cplusplus >= 201402L))')
        h_code_main.putln('#if defined(__cplusplus) && __cplusplus >= 201402L')
        h_code_main.putln('[[deprecated(%s)]] inline' % warning_string.as_c_string_literal())
        h_code_main.putln('#elif defined(__GNUC__) || defined(__clang__)')
        h_code_main.putln('__attribute__ ((__deprecated__(%s), __unused__)) __inline__' % warning_string.as_c_string_literal())
        h_code_main.putln('#elif defined(_MSC_VER)')
        h_code_main.putln('__declspec(deprecated(%s)) __inline' % warning_string.as_c_string_literal())
        h_code_main.putln('#endif')
        h_code_main.putln('static PyObject* __PYX_WARN_IF_%s_INIT_CALLED(PyObject* res) {' % py3_mod_func_name)
        h_code_main.putln('return res;')
        h_code_main.putln('}')
        h_code_main.putln('#define %s() __PYX_WARN_IF_%s_INIT_CALLED(%s())' % (py3_mod_func_name, py3_mod_func_name, py3_mod_func_name))
        h_code_main.putln('#endif')
        h_code_main.putln('#endif')
        h_code_end.putln('')
        h_code_end.putln('#endif /* !%s */' % h_guard)
        with open_new_file(result.h_file) as f:
            h_code_writer.copyto(f)