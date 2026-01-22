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
def test_mod_pow():
    for s, t, u, v in [(4, 13, 497, 445), (4, -3, 497, 365), (3.2, 2.1, 1.9, 0.1031015682350942), (S(3) / 2, 5, S(5) / 6, S(3) / 32)]:
        assert pow(S(s), t, u) == v
        assert pow(S(s), S(t), u) == v
        assert pow(S(s), t, S(u)) == v
        assert pow(S(s), S(t), S(u)) == v
    assert pow(S(2), S(10000000000), S(3)) == 1
    assert pow(x, y, z) == x ** y % z
    raises(TypeError, lambda: pow(S(4), '13', 497))
    raises(TypeError, lambda: pow(S(4), 13, '497'))