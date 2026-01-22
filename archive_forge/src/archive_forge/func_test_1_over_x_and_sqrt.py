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
def test_1_over_x_and_sqrt():
    assert mcode(1 / x) == '1./x'
    assert mcode(x ** (-1)) == mcode(x ** (-1.0)) == '1./x'
    assert mcode(1 / sqrt(x)) == '1./sqrt(x)'
    assert mcode(x ** (-S.Half)) == mcode(x ** (-0.5)) == '1./sqrt(x)'
    assert mcode(sqrt(x)) == 'sqrt(x)'
    assert mcode(x ** S.Half) == mcode(x ** 0.5) == 'sqrt(x)'
    assert mcode(1 / pi) == '1/pi'
    assert mcode(pi ** (-1)) == mcode(pi ** (-1.0)) == '1/pi'
    assert mcode(pi ** (-0.5)) == '1/sqrt(pi)'