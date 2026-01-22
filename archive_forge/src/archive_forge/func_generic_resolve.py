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
def generic_resolve(self, instance, attr):
    if attr in instance.struct:
        return instance.struct[attr]
    elif attr in instance.jit_methods:
        meth = instance.jit_methods[attr]
        disp_type = types.Dispatcher(meth)

        class MethodTemplate(templates.AbstractTemplate):
            key = (self.key, attr)

            def generic(self, args, kws):
                args = (instance,) + tuple(args)
                sig = disp_type.get_call_type(self.context, args, kws)
                return sig.as_method()
        return types.BoundFunction(MethodTemplate, instance)
    elif attr in instance.jit_static_methods:
        meth = instance.jit_static_methods[attr]
        disp_type = types.Dispatcher(meth)

        class StaticMethodTemplate(templates.AbstractTemplate):
            key = (self.key, attr)

            def generic(self, args, kws):
                sig = disp_type.get_call_type(self.context, args, kws)
                return sig.replace(recvr=instance)
        return types.BoundFunction(StaticMethodTemplate, instance)
    elif attr in instance.jit_props:
        impdct = instance.jit_props[attr]
        getter = impdct['get']
        disp_type = types.Dispatcher(getter)
        sig = disp_type.get_call_type(self.context, (instance,), {})
        return sig.return_type