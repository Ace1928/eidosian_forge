import math
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.evalf import N
from sympy.core.function import (Function, nfloat)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio)
from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Rational,
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.complexes import (Abs, re, im)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, cosh)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.printing import srepr
from sympy.printing.str import sstr
from sympy.simplify.simplify import simplify
from sympy.core.numbers import comp
from sympy.core.evalf import (complex_accuracy, PrecisionExhausted,
from mpmath import inf, ninf, make_mpc
from mpmath.libmp.libmpf import from_float, fzero
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import n, x, y
def test_evalf_with_bounded_error():
    cases = [(Rational(0), None, 1), (pi, None, 10), (pi * I, None, 10), (2 - 3 * I, None, 5), (Rational(0), Rational(1, 2), None), (pi, Rational(1, 1000), None), (pi * I, Rational(1, 1000), None), (2 - 3 * I, Rational(1, 1000), None), (2 - 3 * I, Rational(1000), None), (Rational(1234, 10 ** 8), Rational(1, 10 ** 12), None)]
    for x0, eps, m in cases:
        a, b, _, _ = evalf(x0, 53, {})
        c, d, _, _ = _evalf_with_bounded_error(x0, eps, m)
        if eps is None:
            eps = 2 ** (-m)
        z = make_mpc((a or fzero, b or fzero))
        w = make_mpc((c or fzero, d or fzero))
        assert abs(w - z) < eps
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, Rational(0)))
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, -pi))
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, I))