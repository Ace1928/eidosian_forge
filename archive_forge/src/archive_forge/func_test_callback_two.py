from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_callback_two():
    e = 3 * a * b
    f = g.llvm_callable([a, b], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(2)
    array_type = ctypes.c_double * 2
    inp = {a: 0.2, b: 1.7}
    array = array_type(inp[a], inp[b])
    jit_res = f(m, array)
    res = float(e.subs(inp).evalf())
    assert isclose(jit_res, res)