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
def generate_dealloc_function(self, scope, code):
    tp_slot = TypeSlots.ConstructorSlot('tp_dealloc', '__dealloc__')
    slot_func = scope.mangle_internal('tp_dealloc')
    base_type = scope.parent_type.base_type
    if tp_slot.slot_code(scope) != slot_func:
        return
    slot_func_cname = scope.mangle_internal('tp_dealloc')
    code.putln('')
    code.putln('static void %s(PyObject *o) {' % slot_func_cname)
    is_final_type = scope.parent_type.is_final_type
    needs_gc = scope.needs_gc()
    needs_trashcan = scope.needs_trashcan()
    weakref_slot = scope.lookup_here('__weakref__') if not scope.is_closure_class_scope else None
    if weakref_slot not in scope.var_entries:
        weakref_slot = None
    dict_slot = scope.lookup_here('__dict__') if not scope.is_closure_class_scope else None
    if dict_slot not in scope.var_entries:
        dict_slot = None
    _, (py_attrs, _, memoryview_slices) = scope.get_refcounted_entries()
    cpp_destructable_attrs = [entry for entry in scope.var_entries if entry.type.needs_cpp_construction]
    if py_attrs or cpp_destructable_attrs or memoryview_slices or weakref_slot or dict_slot:
        self.generate_self_cast(scope, code)
    if not is_final_type or scope.may_have_finalize():
        code.putln('#if CYTHON_USE_TP_FINALIZE')
        if needs_gc:
            finalised_check = '!__Pyx_PyObject_GC_IsFinalized(o)'
        else:
            finalised_check = '(!PyType_IS_GC(Py_TYPE(o)) || !__Pyx_PyObject_GC_IsFinalized(o))'
        code.putln('if (unlikely((PY_VERSION_HEX >= 0x03080000 || __Pyx_PyType_HasFeature(Py_TYPE(o), Py_TPFLAGS_HAVE_FINALIZE)) && __Pyx_PyObject_GetSlot(o, tp_finalize, destructor)) && %s) {' % finalised_check)
        code.putln('if (__Pyx_PyObject_GetSlot(o, tp_dealloc, destructor) == %s) {' % slot_func_cname)
        code.putln('if (PyObject_CallFinalizerFromDealloc(o)) return;')
        code.putln('}')
        code.putln('}')
        code.putln('#endif')
    if needs_gc:
        code.putln('PyObject_GC_UnTrack(o);')
    if needs_trashcan:
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyTrashcan', 'ExtensionTypes.c'))
        code.putln('__Pyx_TRASHCAN_BEGIN(o, %s)' % slot_func_cname)
    if weakref_slot:
        code.putln('if (p->__weakref__) PyObject_ClearWeakRefs(o);')
    self.generate_usr_dealloc_call(scope, code)
    if dict_slot:
        code.putln('if (p->__dict__) PyDict_Clear(p->__dict__);')
    for entry in cpp_destructable_attrs:
        code.putln('__Pyx_call_destructor(p->%s);' % entry.cname)
    for entry in py_attrs + memoryview_slices:
        code.put_xdecref_clear('p->%s' % entry.cname, entry.type, nanny=False, clear_before_decref=True, have_gil=True)
    if base_type:
        base_cname = base_type.typeptr_cname
        tp_dealloc = TypeSlots.get_base_slot_function(scope, tp_slot)
        if tp_dealloc is not None:
            if needs_gc and base_type.scope and base_type.scope.needs_gc():
                code.putln('PyObject_GC_Track(o);')
            code.putln('%s(o);' % tp_dealloc)
        elif base_type.is_builtin_type:
            if needs_gc and base_type.scope and base_type.scope.needs_gc():
                code.putln('PyObject_GC_Track(o);')
            code.putln('__Pyx_PyType_GetSlot(%s, tp_dealloc, destructor)(o);' % base_cname)
        else:
            if needs_gc:
                code.putln('#if PY_MAJOR_VERSION < 3')
                code.putln('if (!(%s) || PyType_IS_GC(%s)) PyObject_GC_Track(o);' % (base_cname, base_cname))
                code.putln('#else')
                code.putln('if (PyType_IS_GC(%s)) PyObject_GC_Track(o);' % base_cname)
                code.putln('#endif')
            code.putln('if (likely(%s)) __Pyx_PyType_GetSlot(%s, tp_dealloc, destructor)(o); else __Pyx_call_next_tp_dealloc(o, %s);' % (base_cname, base_cname, slot_func_cname))
            code.globalstate.use_utility_code(UtilityCode.load_cached('CallNextTpDealloc', 'ExtensionTypes.c'))
    else:
        freelist_size = scope.directives.get('freelist', 0)
        if freelist_size:
            freelist_name = scope.mangle_internal(Naming.freelist_name)
            freecount_name = scope.mangle_internal(Naming.freecount_name)
            if is_final_type:
                type_safety_check = ''
            else:
                type_safety_check = ' & (int)(!__Pyx_PyType_HasFeature(Py_TYPE(o), (Py_TPFLAGS_IS_ABSTRACT | Py_TPFLAGS_HEAPTYPE)))'
            type = scope.parent_type
            code.putln('#if CYTHON_USE_FREELISTS')
            code.putln('if (((int)(%s < %d) & (int)(Py_TYPE(o)->tp_basicsize == sizeof(%s))%s)) {' % (freecount_name, freelist_size, type.declaration_code('', deref=True), type_safety_check))
            code.putln('%s[%s++] = %s;' % (freelist_name, freecount_name, type.cast_code('o')))
            code.putln('} else')
            code.putln('#endif')
            code.putln('{')
        code.putln('#if CYTHON_USE_TYPE_SLOTS || CYTHON_COMPILING_IN_PYPY')
        code.putln('(*Py_TYPE(o)->tp_free)(o);')
        code.putln('#else')
        code.putln('{')
        code.putln('freefunc tp_free = (freefunc)PyType_GetSlot(Py_TYPE(o), Py_tp_free);')
        code.putln('if (tp_free) tp_free(o);')
        code.putln('}')
        code.putln('#endif')
        if freelist_size:
            code.putln('}')
    if needs_trashcan:
        code.putln('__Pyx_TRASHCAN_END')
    code.putln('}')