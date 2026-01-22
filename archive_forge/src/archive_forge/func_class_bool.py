from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@class_instance_overload(bool)
def class_bool(x):
    using_bool_impl = try_call_method(x, '__bool__')
    if '__len__' in x.jit_methods:

        def using_len_impl(x):
            return bool(len(x))
    else:
        using_len_impl = None
    always_true_impl = lambda x: True
    return take_first(using_bool_impl, using_len_impl, always_true_impl)