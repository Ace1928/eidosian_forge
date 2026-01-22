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
def test_Function_change_name():
    assert mcode(abs(x)) == 'abs(x)'
    assert mcode(ceiling(x)) == 'ceil(x)'
    assert mcode(arg(x)) == 'angle(x)'
    assert mcode(im(x)) == 'imag(x)'
    assert mcode(re(x)) == 'real(x)'
    assert mcode(conjugate(x)) == 'conj(x)'
    assert mcode(chebyshevt(y, x)) == 'chebyshevT(y, x)'
    assert mcode(chebyshevu(y, x)) == 'chebyshevU(y, x)'
    assert mcode(laguerre(x, y)) == 'laguerreL(x, y)'
    assert mcode(Chi(x)) == 'coshint(x)'
    assert mcode(Shi(x)) == 'sinhint(x)'
    assert mcode(Ci(x)) == 'cosint(x)'
    assert mcode(Si(x)) == 'sinint(x)'
    assert mcode(li(x)) == 'logint(x)'
    assert mcode(loggamma(x)) == 'gammaln(x)'
    assert mcode(polygamma(x, y)) == 'psi(x, y)'
    assert mcode(RisingFactorial(x, y)) == 'pochhammer(x, y)'
    assert mcode(DiracDelta(x)) == 'dirac(x)'
    assert mcode(DiracDelta(x, 3)) == 'dirac(3, x)'
    assert mcode(Heaviside(x)) == 'heaviside(x, 1/2)'
    assert mcode(Heaviside(x, y)) == 'heaviside(x, y)'
    assert mcode(binomial(x, y)) == 'bincoeff(x, y)'
    assert mcode(Mod(x, y)) == 'mod(x, y)'