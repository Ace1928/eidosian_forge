from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_simple_rust_code():
    name_expr = ('test', (x + y) * z)
    result, = codegen(name_expr, 'Rust', header=False, empty=False)
    assert result[0] == 'test.rs'
    source = result[1]
    expected = 'fn test(x: f64, y: f64, z: f64) -> f64 {\n    let out1 = z*(x + y);\n    out1\n}\n'
    assert source == expected