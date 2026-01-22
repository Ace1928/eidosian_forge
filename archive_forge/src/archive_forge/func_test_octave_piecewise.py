from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.functions import (arg, atan2, bernoulli, beta, ceiling, chebyshevu,
from sympy.functions import (sin, cos, tan, cot, sec, csc, asin, acos, acot,
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.functions.special.gamma_functions import (gamma, lowergamma,
from sympy.functions.special.error_functions import (Chi, Ci, erf, erfc, erfi,
from sympy.printing.octave import octave_code, octave_code as mcode
def test_octave_piecewise():
    expr = Piecewise((x, x < 1), (x ** 2, True))
    assert mcode(expr) == '((x < 1).*(x) + (~(x < 1)).*(x.^2))'
    assert mcode(expr, assign_to='r') == 'r = ((x < 1).*(x) + (~(x < 1)).*(x.^2));'
    assert mcode(expr, assign_to='r', inline=False) == 'if (x < 1)\n  r = x;\nelse\n  r = x.^2;\nend'
    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = '((x < 1).*(x.^2) + (~(x < 1)).*( ...\n(x < 2).*(x.^3) + (~(x < 2)).*( ...\n(x < 3).*(x.^4) + (~(x < 3)).*(x.^5))))'
    assert mcode(expr) == expected
    assert mcode(expr, assign_to='r') == 'r = ' + expected + ';'
    assert mcode(expr, assign_to='r', inline=False) == 'if (x < 1)\n  r = x.^2;\nelseif (x < 2)\n  r = x.^3;\nelseif (x < 3)\n  r = x.^4;\nelse\n  r = x.^5;\nend'
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: mcode(expr))