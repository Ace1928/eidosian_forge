from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, pi, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, Or, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.parsing.sympy_parser import null
from sympy.polys.polytools import Poly
from sympy.printing.repr import srepr
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.abc import x, y
from sympy.core.sympify import (sympify, _sympify, SympifyError, kernS,
from sympy.core.decorators import _sympifyit
from sympy.external import import_module
from sympy.testing.pytest import raises, XFAIL, skip, warns_deprecated_sympy
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.geometry import Point, Line
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.abc import _clash, _clash1, _clash2
from sympy.external.gmpy import HAS_GMPY
from sympy.sets import FiniteSet, EmptySet
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
import mpmath
from collections import defaultdict, OrderedDict
from mpmath.rational import mpq
def test_sympify1():
    assert sympify('x') == Symbol('x')
    assert sympify('   x') == Symbol('x')
    assert sympify('   x   ') == Symbol('x')
    assert sympify('--.5') == 0.5
    assert sympify('-1/2') == -S.Half
    assert sympify('-+--.5') == -0.5
    assert sympify('-.[3]') == Rational(-1, 3)
    assert sympify('.[3]') == Rational(1, 3)
    assert sympify('+.[3]') == Rational(1, 3)
    assert sympify('+0.[3]*10**-2') == Rational(1, 300)
    assert sympify('.[052631578947368421]') == Rational(1, 19)
    assert sympify('.0[526315789473684210]') == Rational(1, 19)
    assert sympify('.034[56]') == Rational(1711, 49500)
    assert sympify('1.22[345]', rational=True) == 1 + Rational(22, 100) + Rational(345, 99900)
    assert sympify('2/2.6', rational=True) == Rational(10, 13)
    assert sympify('2.6/2', rational=True) == Rational(13, 10)
    assert sympify('2.6e2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e+2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e-2/17', rational=True) == Rational(26, 17000)
    assert sympify('2.1+3/4', rational=True) == Rational(21, 10) + Rational(3, 4)
    assert sympify('2.234456', rational=True) == Rational(279307, 125000)
    assert sympify('2.234456e23', rational=True) == 223445600000000000000000
    assert sympify('2.234456e-23', rational=True) == Rational(279307, 12500000000000000000000000000)
    assert sympify('-2.234456e-23', rational=True) == Rational(-279307, 12500000000000000000000000000)
    assert sympify('12345678901/17', rational=True) == Rational(12345678901, 17)
    assert sympify('1/.3 + x', rational=True) == Rational(10, 3) + x
    assert sympify('222222222222/11111111111') == Rational(222222222222, 11111111111)
    assert sympify('1/.2[123456789012]') == Rational(333333333333, 70781892967)
    assert sympify('.1234567890123456', rational=True) == Rational(19290123283179, 156250000000000)