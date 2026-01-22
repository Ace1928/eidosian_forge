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
def test_Pow_is_even_odd():
    x = Symbol('x')
    k = Symbol('k', even=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', integer=True, nonnegative=True)
    p = Symbol('p', integer=True, positive=True)
    assert ((-1) ** n).is_odd
    assert ((-1) ** k).is_odd
    assert ((-1) ** (m - p)).is_odd
    assert (k ** 2).is_even is True
    assert (n ** 2).is_even is False
    assert (2 ** k).is_even is None
    assert (x ** 2).is_even is None
    assert (k ** m).is_even is None
    assert (n ** m).is_even is False
    assert (k ** p).is_even is True
    assert (n ** p).is_even is False
    assert (m ** k).is_even is None
    assert (p ** k).is_even is None
    assert (m ** n).is_even is None
    assert (p ** n).is_even is None
    assert (k ** x).is_even is None
    assert (n ** x).is_even is None
    assert (k ** 2).is_odd is False
    assert (n ** 2).is_odd is True
    assert (3 ** k).is_odd is None
    assert (k ** m).is_odd is None
    assert (n ** m).is_odd is True
    assert (k ** p).is_odd is False
    assert (n ** p).is_odd is True
    assert (m ** k).is_odd is None
    assert (p ** k).is_odd is None
    assert (m ** n).is_odd is None
    assert (p ** n).is_odd is None
    assert (k ** x).is_odd is None
    assert (n ** x).is_odd is None