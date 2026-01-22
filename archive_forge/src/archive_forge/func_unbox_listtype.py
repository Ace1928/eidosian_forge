from collections.abc import MutableSequence
from numba.core.types import ListType
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import (
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
@unbox(types.ListType)
def unbox_listtype(typ, val, c):
    context = c.context
    builder = c.builder
    list_type = c.pyapi.unserialize(c.pyapi.serialize_object(List))
    valtype = c.pyapi.object_type(val)
    same_type = builder.icmp_unsigned('==', valtype, list_type)
    with c.builder.if_else(same_type) as (then, orelse):
        with then:
            miptr = c.pyapi.object_getattr_string(val, '_opaque')
            native = c.unbox(types.MemInfoPointer(types.voidptr), miptr)
            mi = native.value
            ctor = cgutils.create_struct_proxy(typ)
            lstruct = ctor(context, builder)
            data_pointer = context.nrt.meminfo_data(builder, mi)
            data_pointer = builder.bitcast(data_pointer, listobject.ll_list_type.as_pointer())
            lstruct.data = builder.load(data_pointer)
            lstruct.meminfo = mi
            lstobj = lstruct._getvalue()
            c.pyapi.decref(miptr)
            bb_unboxed = c.builder.basic_block
        with orelse:
            c.pyapi.err_format('PyExc_TypeError', "can't unbox a %S as a %S", valtype, list_type)
            bb_else = c.builder.basic_block
    lstobj_res = c.builder.phi(lstobj.type)
    is_error_res = c.builder.phi(cgutils.bool_t)
    lstobj_res.add_incoming(lstobj, bb_unboxed)
    lstobj_res.add_incoming(lstobj.type(None), bb_else)
    is_error_res.add_incoming(cgutils.false_bit, bb_unboxed)
    is_error_res.add_incoming(cgutils.true_bit, bb_else)
    c.pyapi.decref(list_type)
    c.pyapi.decref(valtype)
    return NativeValue(lstobj_res, is_error=is_error_res)