from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.NumPyRandomBitGeneratorType)
def unbox_numpy_random_bitgenerator(typ, obj, c):
    """
    The bit_generator instance has a `.ctypes` attr which is a namedtuple
    with the following members (types):
    * state_address (Python int)
    * state (ctypes.c_void_p)
    * next_uint64 (ctypes.CFunctionType instance)
    * next_uint32 (ctypes.CFunctionType instance)
    * next_double (ctypes.CFunctionType instance)
    * bit_generator (ctypes.c_void_p)
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    extra_refs = []

    def clear_extra_refs():
        for _ref in extra_refs:
            c.pyapi.decref(_ref)

    def handle_failure():
        c.builder.store(cgutils.true_bit, is_error_ptr)
        clear_extra_refs()
    with ExitStack() as stack:

        def object_getattr_safely(obj, attr):
            attr_obj = c.pyapi.object_getattr_string(obj, attr)
            extra_refs.append(attr_obj)
            return attr_obj
        struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        struct_ptr.parent = obj
        ctypes_binding = object_getattr_safely(obj, 'ctypes')
        with cgutils.early_exit_if_null(c.builder, stack, ctypes_binding):
            handle_failure()
        interface_state_address = object_getattr_safely(ctypes_binding, 'state_address')
        with cgutils.early_exit_if_null(c.builder, stack, interface_state_address):
            handle_failure()
        setattr(struct_ptr, 'state_address', c.unbox(types.uintp, interface_state_address).value)
        interface_state = object_getattr_safely(ctypes_binding, 'state')
        with cgutils.early_exit_if_null(c.builder, stack, interface_state):
            handle_failure()
        interface_state_value = object_getattr_safely(interface_state, 'value')
        with cgutils.early_exit_if_null(c.builder, stack, interface_state_value):
            handle_failure()
        setattr(struct_ptr, 'state', c.unbox(types.uintp, interface_state_value).value)
        ctypes_name = c.context.insert_const_string(c.builder.module, 'ctypes')
        ctypes_module = c.pyapi.import_module_noblock(ctypes_name)
        extra_refs.append(ctypes_module)
        with cgutils.early_exit_if_null(c.builder, stack, ctypes_module):
            handle_failure()
        ct_cast = object_getattr_safely(ctypes_module, 'cast')
        with cgutils.early_exit_if_null(c.builder, stack, ct_cast):
            handle_failure()
        ct_voidptr_ty = object_getattr_safely(ctypes_module, 'c_void_p')
        with cgutils.early_exit_if_null(c.builder, stack, ct_voidptr_ty):
            handle_failure()

        def wire_in_fnptrs(name):
            interface_next_fn = c.pyapi.object_getattr_string(ctypes_binding, name)
            extra_refs.append(interface_next_fn)
            with cgutils.early_exit_if_null(c.builder, stack, interface_next_fn):
                handle_failure()
            args = c.pyapi.tuple_pack([interface_next_fn, ct_voidptr_ty])
            with cgutils.early_exit_if_null(c.builder, stack, args):
                handle_failure()
            interface_next_fn_casted = c.pyapi.call(ct_cast, args)
            interface_next_fn_casted_value = object_getattr_safely(interface_next_fn_casted, 'value')
            with cgutils.early_exit_if_null(c.builder, stack, interface_next_fn_casted_value):
                handle_failure()
            setattr(struct_ptr, f'fnptr_{name}', c.unbox(types.uintp, interface_next_fn_casted_value).value)
        wire_in_fnptrs('next_double')
        wire_in_fnptrs('next_uint64')
        wire_in_fnptrs('next_uint32')
        clear_extra_refs()
    return NativeValue(struct_ptr._getvalue(), is_error=c.builder.load(is_error_ptr))