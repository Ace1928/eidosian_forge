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
def test_ncmul():
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)
    C = Symbol('C', commutative=False)
    assert A * B != B * A
    assert A * B * C != C * B * A
    assert A * b * B * 3 * C == 3 * b * A * B * C
    assert A * b * B * 3 * C != 3 * b * B * A * C
    assert A * b * B * 3 * C == 3 * A * B * C * b
    assert A + B == B + A
    assert (A + B) * C != C * (A + B)
    assert C * (A + B) * C != C * C * (A + B)
    assert A * A == A ** 2
    assert (A + B) * (A + B) == (A + B) ** 2
    assert A ** (-1) * A == 1
    assert A / A == 1
    assert A / A ** 2 == 1 / A
    assert A / (1 + A) == A / (1 + A)
    assert set((A + B + 2 * (A + B)).args) == {A, B, 2 * (A + B)}