from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet
from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin
from itertools import product
def test_Mul_doesnt_expand_exp():
    x = Symbol('x')
    y = Symbol('y')
    assert unchanged(Mul, exp(x), exp(y))
    assert unchanged(Mul, 2 ** x, 2 ** y)
    assert x ** 2 * x ** 3 == x ** 5
    assert 2 ** x * 3 ** x == 6 ** x
    assert x ** y * x ** (2 * y) == x ** (3 * y)
    assert sqrt(2) * sqrt(2) == 2
    assert 2 ** x * 2 ** (2 * x) == 2 ** (3 * x)
    assert sqrt(2) * 2 ** Rational(1, 4) * 5 ** Rational(3, 4) == 10 ** Rational(3, 4)
    assert x ** (-log(5) / log(3)) * x / (x * x ** (-log(5) / log(3))) == sympify(1)