from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_callback_cubature():
    e = a + 1.2
    f = g.llvm_callable([a], e, callback_type='cubature')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    inp = {a: 2.2}
    array = array_type(inp[a])
    out_array = array_type(0.0)
    jit_ret = f(m, array, None, m, out_array)
    assert jit_ret == 0
    res = float(e.subs(inp).evalf())
    assert isclose(out_array[0], res)