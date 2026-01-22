from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_piecewise_():
    pw = Piecewise((0, x < -1), (x ** 2, x <= 1), (-x + 2, x > 1), (1, True), evaluate=False)
    name_expr = ('pwtest', pw)
    result, = codegen(name_expr, 'Rust', header=False, empty=False)
    source = result[1]
    expected = 'fn pwtest(x: f64) -> f64 {\n    let out1 = if (x < -1) {\n        0\n    } else if (x <= 1) {\n        x.powi(2)\n    } else if (x > 1) {\n        2 - x\n    } else {\n        1\n    };\n    out1\n}\n'
    assert source == expected