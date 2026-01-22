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
def generate_c_code(self, env, options, result):
    self.assure_safe_target(result.c_file, allow_failed=True)
    modules = self.referenced_modules
    if Options.annotate or options.annotate:
        show_entire_c_code = Options.annotate == 'fullc' or options.annotate == 'fullc'
        rootwriter = Annotate.AnnotationCCodeWriter(show_entire_c_code=show_entire_c_code, source_desc=self.compilation_source.source_desc)
    else:
        rootwriter = Code.CCodeWriter()
    c_code_config = generate_c_code_config(env, options)
    globalstate = Code.GlobalState(rootwriter, self, code_config=c_code_config, common_utility_include_dir=options.common_utility_include_dir)
    globalstate.initialize_main_c_code()
    h_code = globalstate['h_code']
    self.generate_module_preamble(env, options, modules, result.embedded_metadata, h_code)
    globalstate.module_pos = self.pos
    globalstate.directives = self.directives
    globalstate.use_utility_code(refnanny_utility_code)
    code = globalstate['before_global_var']
    code.putln('#define __Pyx_MODULE_NAME %s' % self.full_module_name.as_c_string_literal())
    module_is_main = self.is_main_module_flag_cname()
    code.putln('extern int %s;' % module_is_main)
    code.putln('int %s = 0;' % module_is_main)
    code.putln('')
    code.putln('/* Implementation of %s */' % env.qualified_name.as_c_string_literal())
    code = globalstate['late_includes']
    self.generate_includes(env, modules, code, early=False)
    code = globalstate['module_code']
    self.generate_cached_builtins_decls(env, code)
    self.generate_lambda_definitions(env, code)
    self.generate_variable_definitions(env, code)
    self.body.generate_function_definitions(env, code)
    code.mark_pos(None)
    self.generate_typeobj_definitions(env, code)
    self.generate_method_table(env, code)
    if env.has_import_star:
        self.generate_import_star(env, code)
    code.putln(UtilityCode.load_as_string('SmallCodeConfig', 'ModuleSetupCode.c')[0].strip())
    self.generate_module_state_start(env, globalstate['module_state'])
    self.generate_module_state_defines(env, globalstate['module_state_defines'])
    self.generate_module_state_clear(env, globalstate['module_state_clear'])
    self.generate_module_state_traverse(env, globalstate['module_state_traverse'])
    self.generate_module_init_func(modules[:-1], env, globalstate['init_module'])
    self.generate_module_cleanup_func(env, globalstate['cleanup_module'])
    if Options.embed:
        self.generate_main_method(env, globalstate['main_method'])
    self.generate_filename_table(globalstate['filename_table'])
    self.generate_declarations_for_modules(env, modules, globalstate)
    h_code.write('\n')
    for utilcode in env.utility_code_list[:]:
        globalstate.use_utility_code(utilcode)
    globalstate.finalize_main_c_code()
    self.generate_module_state_end(env, modules, globalstate)
    f = open_new_file(result.c_file)
    try:
        rootwriter.copyto(f)
    finally:
        f.close()
    result.c_file_generated = 1
    if options.gdb_debug:
        self._serialize_lineno_map(env, rootwriter)
    if Options.annotate or options.annotate:
        self._generate_annotations(rootwriter, result, options)