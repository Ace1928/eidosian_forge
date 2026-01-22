from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_not_supported():
    f = Function('f')
    name_expr = ('test', [f(x).diff(x), S.ComplexInfinity])
    result, = codegen(name_expr, 'Rust', header=False, empty=False)
    source = result[1]
    expected = 'fn test(x: f64) -> (f64, f64) {\n    // unsupported: Derivative(f(x), x)\n    // unsupported: zoo\n    let out1 = Derivative(f(x), x);\n    let out2 = zoo;\n    (out1, out2)\n}\n'
    assert source == expected