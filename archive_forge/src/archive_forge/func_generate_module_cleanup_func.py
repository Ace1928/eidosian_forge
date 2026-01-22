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
def generate_module_cleanup_func(self, env, code):
    if not Options.generate_cleanup_code:
        return
    code.putln('static void %s(CYTHON_UNUSED PyObject *self) {' % Naming.cleanup_cname)
    code.enter_cfunc_scope(env)
    if Options.generate_cleanup_code >= 2:
        code.putln('/*--- Global cleanup code ---*/')
        rev_entries = list(env.var_entries)
        rev_entries.reverse()
        for entry in rev_entries:
            if entry.visibility != 'extern':
                if entry.type.is_pyobject and entry.used:
                    code.put_xdecref_clear(entry.cname, entry.type, clear_before_decref=True, nanny=False)
    code.putln('__Pyx_CleanupGlobals();')
    if Options.generate_cleanup_code >= 3:
        code.putln('/*--- Type import cleanup code ---*/')
        for ext_type in sorted(env.types_imported, key=operator.attrgetter('typeptr_cname')):
            code.put_xdecref_clear(ext_type.typeptr_cname, ext_type, clear_before_decref=True, nanny=False)
    if Options.cache_builtins:
        code.putln('/*--- Builtin cleanup code ---*/')
        for entry in env.cached_builtins:
            code.put_xdecref_clear(entry.cname, PyrexTypes.py_object_type, clear_before_decref=True, nanny=False)
    code.putln('/*--- Intern cleanup code ---*/')
    code.put_decref_clear(Naming.empty_tuple, PyrexTypes.py_object_type, clear_before_decref=True, nanny=False)
    for entry in env.c_class_entries:
        cclass_type = entry.type
        if cclass_type.is_external or cclass_type.base_type:
            continue
        if cclass_type.scope.directives.get('freelist', 0):
            scope = cclass_type.scope
            freelist_name = scope.mangle_internal(Naming.freelist_name)
            freecount_name = scope.mangle_internal(Naming.freecount_name)
            code.putln('#if CYTHON_USE_FREELISTS')
            code.putln('while (%s > 0) {' % freecount_name)
            code.putln('PyObject* o = (PyObject*)%s[--%s];' % (freelist_name, freecount_name))
            code.putln('#if CYTHON_USE_TYPE_SLOTS || CYTHON_COMPILING_IN_PYPY')
            code.putln('(*Py_TYPE(o)->tp_free)(o);')
            code.putln('#else')
            code.putln('freefunc tp_free = (freefunc)PyType_GetSlot(Py_TYPE(o), Py_tp_free);')
            code.putln('if (tp_free) tp_free(o);')
            code.putln('#endif')
            code.putln('}')
            code.putln('#endif')
    if Options.pre_import is not None:
        code.put_decref_clear(Naming.preimport_cname, py_object_type, nanny=False, clear_before_decref=True)
    for cname in [Naming.cython_runtime_cname, Naming.builtins_cname]:
        code.put_decref_clear(cname, py_object_type, nanny=False, clear_before_decref=True)
    code.put_decref_clear(env.module_dict_cname, py_object_type, nanny=False, clear_before_decref=True)