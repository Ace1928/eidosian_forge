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
def test_comparisons_with_unknown_type():

    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.

        """
    ni, nf, nr = (Integer(3), Float(1.0), Rational(1, 3))
    foo = Foo()
    for n in (ni, nf, nr, oo, -oo, zoo, nan):
        assert n != foo
        assert foo != n
        assert not n == foo
        assert not foo == n
        raises(TypeError, lambda: n < foo)
        raises(TypeError, lambda: foo > n)
        raises(TypeError, lambda: n > foo)
        raises(TypeError, lambda: foo < n)
        raises(TypeError, lambda: n <= foo)
        raises(TypeError, lambda: foo >= n)
        raises(TypeError, lambda: n >= foo)
        raises(TypeError, lambda: foo <= n)

    class Bar:
        """
        Class that considers itself equal to any instance of Number except
        infinities and nans, and relies on SymPy types returning the
        NotImplemented singleton for symmetric equality relations.

        """

        def __eq__(self, other):
            if other in (oo, -oo, zoo, nan):
                return False
            if isinstance(other, Number):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other
    bar = Bar()
    for n in (ni, nf, nr):
        assert n == bar
        assert bar == n
        assert not n != bar
        assert not bar != n
    for n in (oo, -oo, zoo, nan):
        assert n != bar
        assert bar != n
        assert not n == bar
        assert not bar == n
    for n in (ni, nf, nr, oo, -oo, zoo, nan):
        raises(TypeError, lambda: n < bar)
        raises(TypeError, lambda: bar > n)
        raises(TypeError, lambda: n > bar)
        raises(TypeError, lambda: bar < n)
        raises(TypeError, lambda: n <= bar)
        raises(TypeError, lambda: bar >= n)
        raises(TypeError, lambda: n >= bar)
        raises(TypeError, lambda: bar <= n)