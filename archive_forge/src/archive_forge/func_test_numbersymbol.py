from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_numbersymbol():
    name_expr = ('test', pi ** Catalan)
    result, = codegen(name_expr, 'Rust', header=False, empty=False)
    source = result[1]
    expected = 'fn test() -> f64 {\n    const Catalan: f64 = %s;\n    let out1 = PI.powf(Catalan);\n    out1\n}\n' % Catalan.evalf(17)
    assert source == expected