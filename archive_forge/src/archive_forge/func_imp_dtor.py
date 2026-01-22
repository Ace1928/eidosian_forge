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
def imp_dtor(context, module, instance_type):
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = llvmir.FunctionType(llvmir.VoidType(), [llvoidptr, llsize, llvoidptr])
    fname = '_Dtor.{0}'.format(instance_type.name)
    dtor_fn = cgutils.get_or_insert_function(module, dtor_ftype, fname)
    if dtor_fn.is_declaration:
        builder = llvmir.IRBuilder(dtor_fn.append_basic_block())
        alloc_fe_type = instance_type.get_data_type()
        alloc_type = context.get_value_type(alloc_fe_type)
        ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())
        data = context.make_helper(builder, alloc_fe_type, ref=ptr)
        context.nrt.decref(builder, alloc_fe_type, data._getvalue())
        builder.ret_void()
    return dtor_fn