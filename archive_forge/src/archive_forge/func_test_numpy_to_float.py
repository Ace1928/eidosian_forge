import numbers as nums
import decimal
from sympy.concrete.summations import Sum
from sympy.core import (EulerGamma, Catalan, TribonacciConstant,
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import (mpf_norm, mod_inverse, igcd, seterr,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.integers import floor
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.polys.domains.realfield import RealField
from sympy.printing.latex import latex
from sympy.printing.repr import srepr
from sympy.simplify import simplify
from sympy.core.power import integer_nthroot, isqrt, integer_log
from sympy.polys.domains.groundtypes import PythonRational
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
from mpmath import mpf
from mpmath.rational import mpq
import mpmath
from sympy.core import numbers
def test_numpy_to_float():
    from sympy.testing.pytest import skip
    from sympy.external import import_module
    np = import_module('numpy')
    if not np:
        skip('numpy not installed. Abort numpy tests.')

    def check_prec_and_relerr(npval, ratval):
        prec = np.finfo(npval).nmant + 1
        x = Float(npval)
        assert x._prec == prec
        y = Float(ratval, precision=prec)
        assert abs((x - y) / y) < 2 ** (-(prec + 1))
    check_prec_and_relerr(np.float16(2.0 / 3), Rational(2, 3))
    check_prec_and_relerr(np.float32(2.0 / 3), Rational(2, 3))
    check_prec_and_relerr(np.float64(2.0 / 3), Rational(2, 3))
    x = np.longdouble(2) / 3
    check_prec_and_relerr(x, Rational(2, 3))
    y = Float(x, precision=10)
    assert same_and_same_prec(y, Float(Rational(2, 3), precision=10))
    raises(TypeError, lambda: Float(np.complex64(1 + 2j)))
    raises(TypeError, lambda: Float(np.complex128(1 + 2j)))