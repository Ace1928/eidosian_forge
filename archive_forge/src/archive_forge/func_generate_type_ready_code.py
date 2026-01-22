from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
@staticmethod
def generate_type_ready_code(entry, code, bases_tuple_cname=None, check_heap_type_bases=False):
    type = entry.type
    typeptr_cname = type.typeptr_cname
    scope = type.scope
    if not scope:
        return
    if entry.visibility == 'extern':
        if type.typeobj_cname:
            assert not type.typeobj_cname
            code.putln('%s = &%s;' % (type.typeptr_cname, type.typeobj_cname))
        return
    else:
        assert typeptr_cname
        assert type.typeobj_cname
        typespec_cname = '%s_spec' % type.typeobj_cname
        code.putln('#if CYTHON_USE_TYPE_SPECS')
        tuple_temp = None
        if not bases_tuple_cname and scope.parent_type.base_type:
            tuple_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
            code.putln('%s = PyTuple_Pack(1, (PyObject *)%s); %s' % (tuple_temp, scope.parent_type.base_type.typeptr_cname, code.error_goto_if_null(tuple_temp, entry.pos)))
            code.put_gotref(tuple_temp, py_object_type)
        if bases_tuple_cname or tuple_temp:
            if check_heap_type_bases:
                code.globalstate.use_utility_code(UtilityCode.load_cached('ValidateBasesTuple', 'ExtensionTypes.c'))
                code.put_error_if_neg(entry.pos, '__Pyx_validate_bases_tuple(%s.name, %s, %s)' % (typespec_cname, TypeSlots.get_slot_by_name('tp_dictoffset', scope.directives).slot_code(scope), bases_tuple_cname or tuple_temp))
            code.putln('%s = (PyTypeObject *) __Pyx_PyType_FromModuleAndSpec(%s, &%s, %s);' % (typeptr_cname, Naming.module_cname, typespec_cname, bases_tuple_cname or tuple_temp))
            if tuple_temp:
                code.put_xdecref_clear(tuple_temp, type=py_object_type)
                code.funcstate.release_temp(tuple_temp)
            code.putln(code.error_goto_if_null(typeptr_cname, entry.pos))
        else:
            code.putln('%s = (PyTypeObject *) __Pyx_PyType_FromModuleAndSpec(%s, &%s, NULL); %s' % (typeptr_cname, Naming.module_cname, typespec_cname, code.error_goto_if_null(typeptr_cname, entry.pos)))
        buffer_slot = TypeSlots.get_slot_by_name('tp_as_buffer', code.globalstate.directives)
        if not buffer_slot.is_empty(scope):
            code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
            code.putln('%s->%s = %s;' % (typeptr_cname, buffer_slot.slot_name, buffer_slot.slot_code(scope)))
            for buffer_method_name in ('__getbuffer__', '__releasebuffer__'):
                buffer_slot = TypeSlots.get_slot_table(code.globalstate.directives).get_slot_by_method_name(buffer_method_name)
                if buffer_slot.slot_code(scope) == '0' and (not TypeSlots.get_base_slot_function(scope, buffer_slot)):
                    code.putln('if (!%s->tp_as_buffer->%s && %s->tp_base->tp_as_buffer && %s->tp_base->tp_as_buffer->%s) {' % (typeptr_cname, buffer_slot.slot_name, typeptr_cname, typeptr_cname, buffer_slot.slot_name))
                    code.putln('%s->tp_as_buffer->%s = %s->tp_base->tp_as_buffer->%s;' % (typeptr_cname, buffer_slot.slot_name, typeptr_cname, buffer_slot.slot_name))
                    code.putln('}')
            code.putln('#elif defined(Py_bf_getbuffer) && defined(Py_bf_releasebuffer)')
            code.putln('/* PY_VERSION_HEX >= 0x03090000 || Py_LIMITED_API >= 0x030B0000 */')
            code.putln('#elif defined(_MSC_VER)')
            code.putln('#pragma message ("The buffer protocol is not supported in the Limited C-API < 3.11.")')
            code.putln('#else')
            code.putln('#warning "The buffer protocol is not supported in the Limited C-API < 3.11."')
            code.putln('#endif')
        code.globalstate.use_utility_code(UtilityCode.load_cached('FixUpExtensionType', 'ExtensionTypes.c'))
        code.put_error_if_neg(entry.pos, '__Pyx_fix_up_extension_type_from_spec(&%s, %s)' % (typespec_cname, typeptr_cname))
        code.putln('#else')
        if bases_tuple_cname:
            code.put_incref(bases_tuple_cname, py_object_type)
            code.put_giveref(bases_tuple_cname, py_object_type)
            code.putln('%s.tp_bases = %s;' % (type.typeobj_cname, bases_tuple_cname))
        code.putln('%s = &%s;' % (typeptr_cname, type.typeobj_cname))
        code.putln('#endif')
        base_type = type.base_type
        while base_type:
            if base_type.is_external and (not base_type.objstruct_cname == 'PyTypeObject'):
                code.putln('if (sizeof(%s%s) != sizeof(%s%s)) {' % ('' if type.typedef_flag else 'struct ', type.objstruct_cname, '' if base_type.typedef_flag else 'struct ', base_type.objstruct_cname))
                code.globalstate.use_utility_code(UtilityCode.load_cached('ValidateExternBase', 'ExtensionTypes.c'))
                code.put_error_if_neg(entry.pos, '__Pyx_validate_extern_base(%s)' % type.base_type.typeptr_cname)
                code.putln('}')
                break
            base_type = base_type.base_type
        code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
        for slot in TypeSlots.get_slot_table(code.globalstate.directives):
            slot.generate_dynamic_init_code(scope, code)
        code.putln('#endif')
        code.putln('#if !CYTHON_USE_TYPE_SPECS')
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyType_Ready', 'ExtensionTypes.c'))
        code.put_error_if_neg(entry.pos, '__Pyx_PyType_Ready(%s)' % typeptr_cname)
        code.putln('#endif')
        code.putln('#if PY_MAJOR_VERSION < 3')
        code.putln('%s->tp_print = 0;' % typeptr_cname)
        code.putln('#endif')
        getattr_slot_func = TypeSlots.get_slot_code_by_name(scope, 'tp_getattro')
        dictoffset_slot_func = TypeSlots.get_slot_code_by_name(scope, 'tp_dictoffset')
        if getattr_slot_func == '0' and dictoffset_slot_func == '0':
            code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
            if type.is_final_type:
                py_cfunc = '__Pyx_PyObject_GenericGetAttrNoDict'
                utility_func = 'PyObject_GenericGetAttrNoDict'
            else:
                py_cfunc = '__Pyx_PyObject_GenericGetAttr'
                utility_func = 'PyObject_GenericGetAttr'
            code.globalstate.use_utility_code(UtilityCode.load_cached(utility_func, 'ObjectHandling.c'))
            code.putln('if ((CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP) && likely(!%s->tp_dictoffset && %s->tp_getattro == PyObject_GenericGetAttr)) {' % (typeptr_cname, typeptr_cname))
            code.putln('%s->tp_getattro = %s;' % (typeptr_cname, py_cfunc))
            code.putln('}')
            code.putln('#endif')
        for func in entry.type.scope.pyfunc_entries:
            is_buffer = func.name in ('__getbuffer__', '__releasebuffer__')
            if func.is_special and Options.docstrings and func.wrapperbase_cname and (not is_buffer):
                slot = TypeSlots.get_slot_table(entry.type.scope.directives).get_slot_by_method_name(func.name)
                preprocessor_guard = slot.preprocessor_guard_code() if slot else None
                if preprocessor_guard:
                    code.putln(preprocessor_guard)
                code.putln('#if CYTHON_UPDATE_DESCRIPTOR_DOC')
                code.putln('{')
                code.putln('PyObject *wrapper = PyObject_GetAttrString((PyObject *)%s, "%s"); %s' % (typeptr_cname, func.name, code.error_goto_if_null('wrapper', entry.pos)))
                code.putln('if (__Pyx_IS_TYPE(wrapper, &PyWrapperDescr_Type)) {')
                code.putln('%s = *((PyWrapperDescrObject *)wrapper)->d_base;' % func.wrapperbase_cname)
                code.putln('%s.doc = %s;' % (func.wrapperbase_cname, func.doc_cname))
                code.putln('((PyWrapperDescrObject *)wrapper)->d_base = &%s;' % func.wrapperbase_cname)
                code.putln('}')
                code.putln('}')
                code.putln('#endif')
                if preprocessor_guard:
                    code.putln('#endif')
        if type.vtable_cname:
            code.globalstate.use_utility_code(UtilityCode.load_cached('SetVTable', 'ImportExport.c'))
            code.put_error_if_neg(entry.pos, '__Pyx_SetVtable(%s, %s)' % (typeptr_cname, type.vtabptr_cname))
            code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
            code.globalstate.use_utility_code(UtilityCode.load_cached('MergeVTables', 'ImportExport.c'))
            code.put_error_if_neg(entry.pos, '__Pyx_MergeVtables(%s)' % typeptr_cname)
            code.putln('#endif')
        if not type.scope.is_internal and (not type.scope.directives.get('internal')):
            code.put_error_if_neg(entry.pos, 'PyObject_SetAttr(%s, %s, (PyObject *) %s)' % (Naming.module_cname, code.intern_identifier(scope.class_name), typeptr_cname))
        weakref_entry = scope.lookup_here('__weakref__') if not scope.is_closure_class_scope else None
        if weakref_entry:
            if weakref_entry.type is py_object_type:
                tp_weaklistoffset = '%s->tp_weaklistoffset' % typeptr_cname
                if type.typedef_flag:
                    objstruct = type.objstruct_cname
                else:
                    objstruct = 'struct %s' % type.objstruct_cname
                code.putln('if (%s == 0) %s = offsetof(%s, %s);' % (tp_weaklistoffset, tp_weaklistoffset, objstruct, weakref_entry.cname))
            else:
                error(weakref_entry.pos, "__weakref__ slot must be of type 'object'")
        if scope.lookup_here('__reduce_cython__') if not scope.is_closure_class_scope else None:
            code.globalstate.use_utility_code(UtilityCode.load_cached('SetupReduce', 'ExtensionTypes.c'))
            code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
            code.put_error_if_neg(entry.pos, '__Pyx_setup_reduce((PyObject *) %s)' % typeptr_cname)
            code.putln('#endif')