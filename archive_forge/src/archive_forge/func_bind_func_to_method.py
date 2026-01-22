import types
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_trace_api
def bind_func_to_method(func, obj, method_name):
    bound_method = types.MethodType(func, obj)
    setattr(obj, method_name, bound_method)
    return bound_method