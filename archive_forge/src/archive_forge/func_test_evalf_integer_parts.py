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
def test_evalf_integer_parts():
    a = floor(log(8) / log(2) - exp(-1000), evaluate=False)
    b = floor(log(8) / log(2), evaluate=False)
    assert a.evalf() == 3.0
    assert b.evalf() == 3.0
    assert ceiling(10 * (sin(1) ** 2 + cos(1) ** 2)) == 10
    assert int(floor(factorial(50) / E, evaluate=False).evalf(70)) == int(11188719610782480504630258070757734324011354208865721592720336800)
    assert int(ceiling(factorial(50) / E, evaluate=False).evalf(70)) == int(11188719610782480504630258070757734324011354208865721592720336801)
    assert int(floor(GoldenRatio ** 999 / sqrt(5) + S.Half).evalf(1000)) == fibonacci(999)
    assert int(floor(GoldenRatio ** 1000 / sqrt(5) + S.Half).evalf(1000)) == fibonacci(1000)
    assert ceiling(x).evalf(subs={x: 3}) == 3.0
    assert ceiling(x).evalf(subs={x: 3 * I}) == 3.0 * I
    assert ceiling(x).evalf(subs={x: 2 + 3 * I}) == 2.0 + 3.0 * I
    assert ceiling(x).evalf(subs={x: 3.0}) == 3.0
    assert ceiling(x).evalf(subs={x: 3.0 * I}) == 3.0 * I
    assert ceiling(x).evalf(subs={x: 2.0 + 3 * I}) == 2.0 + 3.0 * I
    assert float((floor(1.5, evaluate=False) + 1 / 9).evalf()) == 1 + 1 / 9
    assert float((floor(0.5, evaluate=False) + 20).evalf()) == 20
    n = 1169809367327212570704813632106852886389036911
    r = 744723773141314414542111064094745678855643068
    assert floor(n / (pi / 2)) == r
    assert floor(80782 * sqrt(2)) == 114242
    assert 260515 - floor(260515 / pi + 1 / 2) * pi == atan(tan(260515))