import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
@ClassBuilder.class_impl_registry.lower(types.ClassType, types.VarArg(types.Any))
def ctor_impl(context, builder, sig, args):
    """
    Generic constructor (__new__) for jitclasses.
    """
    inst_typ = sig.return_type
    alloc_type = context.get_data_type(inst_typ.get_data_type())
    alloc_size = context.get_abi_sizeof(alloc_type)
    meminfo = context.nrt.meminfo_alloc_dtor(builder, context.get_constant(types.uintp, alloc_size), imp_dtor(context, builder.module, inst_typ))
    data_pointer = context.nrt.meminfo_data(builder, meminfo)
    data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())
    builder.store(cgutils.get_null_value(alloc_type), data_pointer)
    inst_struct = context.make_helper(builder, inst_typ)
    inst_struct.meminfo = meminfo
    inst_struct.data = data_pointer
    init_sig = (sig.return_type,) + sig.args
    init = inst_typ.jit_methods['__init__']
    disp_type = types.Dispatcher(init)
    call = context.get_function(disp_type, types.void(*init_sig))
    _add_linking_libs(context, call)
    realargs = [inst_struct._getvalue()] + list(args)
    call(builder, realargs)
    ret = inst_struct._getvalue()
    return imputils.impl_ret_new_ref(context, builder, inst_typ, ret)