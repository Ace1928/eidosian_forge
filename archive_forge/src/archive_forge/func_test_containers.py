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
def test_containers():
    assert mcode([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == '{1, 2, 3, {4, 5, {6, 7}}, 8, {9, 10}, 11}'
    assert mcode((1, 2, (3, 4))) == '{1, 2, {3, 4}}'
    assert mcode([1]) == '{1}'
    assert mcode((1,)) == '{1}'
    assert mcode(Tuple(*[1, 2, 3])) == '{1, 2, 3}'
    assert mcode((1, x * y, (3, x ** 2))) == '{1, x.*y, {3, x.^2}}'
    assert mcode((1, eye(3), Matrix(0, 0, []), [])) == '{1, [1 0 0; 0 1 0; 0 0 1], [], {}}'