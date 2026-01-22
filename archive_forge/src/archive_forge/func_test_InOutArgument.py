from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_InOutArgument():
    expr = Equality(x, x ** 2)
    name_expr = ('mysqr', expr)
    result, = codegen(name_expr, 'Rust', header=False, empty=False)
    source = result[1]
    expected = 'fn mysqr(x: f64) -> f64 {\n    let x = x.powi(2);\n    x\n}\n'
    assert source == expected