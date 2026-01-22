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
def test_issue_6077():
    assert x ** 2.0 / x == x ** 1.0
    assert x / x ** 2.0 == x ** (-1.0)
    assert x * x ** 2.0 == x ** 3.0
    assert x ** 1.5 * x ** 2.5 == x ** 4.0
    assert 2 ** (2.0 * x) / 2 ** x == 2 ** (1.0 * x)
    assert 2 ** x / 2 ** (2.0 * x) == 2 ** (-1.0 * x)
    assert 2 ** x * 2 ** (2.0 * x) == 2 ** (3.0 * x)
    assert 2 ** (1.5 * x) * 2 ** (2.5 * x) == 2 ** (4.0 * x)