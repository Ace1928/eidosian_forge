from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_multifcns_per_file():
    name_expr = [('foo', [2 * x, 3 * y]), ('bar', [y ** 2, 4 * y])]
    result = codegen(name_expr, 'Rust', header=False, empty=False)
    assert result[0][0] == 'foo.rs'
    source = result[0][1]
    expected = 'fn foo(x: f64, y: f64) -> (f64, f64) {\n    let out1 = 2*x;\n    let out2 = 3*y;\n    (out1, out2)\n}\nfn bar(y: f64) -> (f64, f64) {\n    let out1 = y.powi(2);\n    let out2 = 4*y;\n    (out1, out2)\n}\n'
    assert source == expected