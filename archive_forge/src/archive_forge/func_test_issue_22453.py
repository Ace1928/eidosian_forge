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
def test_issue_22453():
    from sympy.utilities.iterables import cartes
    e = Symbol('e', extended_positive=True)
    for a, b in cartes(*[[oo, -oo, 3]] * 2):
        if a == b == 3:
            continue
        i = a + I * b
        assert i ** (1 + e) is S.ComplexInfinity
        assert i ** (-e) is S.Zero
        assert unchanged(Pow, i, e)
    assert 1 / (oo + I * oo) is S.Zero
    r, i = [Dummy(infinite=True, extended_real=True) for _ in range(2)]
    assert 1 / (r + I * i) is S.Zero
    assert 1 / (3 + I * i) is S.Zero
    assert 1 / (r + I * 3) is S.Zero