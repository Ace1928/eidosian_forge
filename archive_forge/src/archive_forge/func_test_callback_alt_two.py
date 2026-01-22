from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_callback_alt_two():
    d = sympy.IndexedBase('d')
    e = 3 * d[0] * d[1]
    f = g.llvm_callable([n, d], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(2)
    array_type = ctypes.c_double * 2
    inp = {d[0]: 0.2, d[1]: 1.7}
    array = array_type(inp[d[0]], inp[d[1]])
    jit_res = f(m, array)
    res = float(e.subs(inp).evalf())
    assert isclose(jit_res, res)