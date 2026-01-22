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
@ClassBuilder.class_impl_registry.lower_setattr_generic(types.ClassInstanceType)
def set_attr_impl(context, builder, sig, args, attr):
    """
    Generic setattr() for @jitclass instances.
    """
    typ, valty = sig.args
    target, val = args
    if attr in typ.struct:
        inst = context.make_helper(builder, typ, value=target)
        data_ptr = inst.data
        data = context.make_data_helper(builder, typ.get_data_type(), ref=data_ptr)
        attr_type = typ.struct[attr]
        oldvalue = getattr(data, _mangle_attr(attr))
        setattr(data, _mangle_attr(attr), val)
        context.nrt.incref(builder, attr_type, val)
        context.nrt.decref(builder, attr_type, oldvalue)
    elif attr in typ.jit_props:
        setter = typ.jit_props[attr]['set']
        disp_type = types.Dispatcher(setter)
        sig = disp_type.get_call_type(context.typing_context, (typ, valty), {})
        call = context.get_function(disp_type, sig)
        call(builder, (target, val))
        _add_linking_libs(context, call)
    else:
        raise NotImplementedError('attribute {0!r} not implemented'.format(attr))