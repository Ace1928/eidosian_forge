import ctypes
import unittest
from numba.core import types
from numba.core.extending import intrinsic
from numba import jit, njit
from numba.tests.support import captured_stdout
@intrinsic
def py_call(tyctx):

    def codegen(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        gil = pyapi.gil_ensure()
        num = pyapi.long_from_longlong(context.get_constant(types.intp, 51966))
        kwds = pyapi.dict_pack({'key': num}.items())
        fn_print = pyapi.unserialize(pyapi.serialize_object(callme))
        res = pyapi.call(fn_print, None, kwds)
        pyapi.decref(res)
        pyapi.decref(fn_print)
        pyapi.decref(kwds)
        pyapi.decref(num)
        pyapi.gil_release(gil)
        return res
    return (types.none(), codegen)