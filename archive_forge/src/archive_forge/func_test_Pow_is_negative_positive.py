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
def test_Pow_is_negative_positive():
    r = Symbol('r', real=True)
    k = Symbol('k', integer=True, positive=True)
    n = Symbol('n', even=True)
    m = Symbol('m', odd=True)
    x = Symbol('x')
    assert (2 ** r).is_positive is True
    assert ((-2) ** r).is_positive is None
    assert ((-2) ** n).is_positive is True
    assert ((-2) ** m).is_positive is False
    assert (k ** 2).is_positive is True
    assert (k ** (-2)).is_positive is True
    assert (k ** r).is_positive is True
    assert ((-k) ** r).is_positive is None
    assert ((-k) ** n).is_positive is True
    assert ((-k) ** m).is_positive is False
    assert (2 ** r).is_negative is False
    assert ((-2) ** r).is_negative is None
    assert ((-2) ** n).is_negative is False
    assert ((-2) ** m).is_negative is True
    assert (k ** 2).is_negative is False
    assert (k ** (-2)).is_negative is False
    assert (k ** r).is_negative is False
    assert ((-k) ** r).is_negative is None
    assert ((-k) ** n).is_negative is False
    assert ((-k) ** m).is_negative is True
    assert (2 ** x).is_positive is None
    assert (2 ** x).is_negative is None