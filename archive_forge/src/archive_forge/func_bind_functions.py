import types
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_trace_api
def bind_functions(self, interface, function_factory, arg):
    for name in dir(interface):
        func = function_factory(arg, name)
        if type(func) == types.FunctionType:
            bind_func_to_method(func, self, name)