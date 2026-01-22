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
def generate_clear_function(self, scope, code, cclass_entry):
    tp_slot = TypeSlots.get_slot_by_name('tp_clear', scope.directives)
    slot_func = scope.mangle_internal('tp_clear')
    base_type = scope.parent_type.base_type
    if tp_slot.slot_code(scope) != slot_func:
        return
    have_entries, (py_attrs, py_buffers, memoryview_slices) = scope.get_refcounted_entries(include_gc_simple=False)
    if py_attrs or py_buffers or base_type:
        unused = ''
    else:
        unused = 'CYTHON_UNUSED '
    code.putln('')
    code.putln('static int %s(%sPyObject *o) {' % (slot_func, unused))
    if py_attrs and Options.clear_to_none:
        code.putln('PyObject* tmp;')
    if py_attrs or py_buffers:
        self.generate_self_cast(scope, code)
    if base_type:
        static_call = TypeSlots.get_base_slot_function(scope, tp_slot)
        if static_call:
            code.putln('%s(o);' % static_call)
        elif base_type.is_builtin_type:
            base_cname = base_type.typeptr_cname
            code.putln('if (!%s->tp_clear); else %s->tp_clear(o);' % (base_cname, base_cname))
        else:
            base_cname = base_type.typeptr_cname
            code.putln('if (likely(%s)) { if (%s->tp_clear) %s->tp_clear(o); } else __Pyx_call_next_tp_clear(o, %s);' % (base_cname, base_cname, base_cname, slot_func))
            code.globalstate.use_utility_code(UtilityCode.load_cached('CallNextTpClear', 'ExtensionTypes.c'))
    if Options.clear_to_none:
        for entry in py_attrs:
            name = 'p->%s' % entry.cname
            code.putln('tmp = ((PyObject*)%s);' % name)
            if entry.is_declared_generic:
                code.put_init_to_py_none(name, py_object_type, nanny=False)
            else:
                code.put_init_to_py_none(name, entry.type, nanny=False)
            code.putln('Py_XDECREF(tmp);')
    else:
        for entry in py_attrs:
            code.putln('Py_CLEAR(p->%s);' % entry.cname)
    for entry in py_buffers:
        code.putln('Py_CLEAR(p->%s.obj);' % entry.cname)
    if cclass_entry.cname == '__pyx_memoryviewslice':
        code.putln('__PYX_XCLEAR_MEMVIEW(&p->from_slice, 1);')
    code.putln('return 0;')
    code.putln('}')