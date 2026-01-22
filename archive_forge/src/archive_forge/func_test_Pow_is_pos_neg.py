from sympy.core.mod import Mod
from sympy.core.numbers import (I, oo, pi)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, sin)
from sympy.simplify.simplify import simplify
from sympy.core import Symbol, S, Rational, Integer, Dummy, Wild, Pow
from sympy.core.assumptions import (assumptions, check_assumptions,
from sympy.core.facts import InconsistentAssumptions
from sympy.core.random import seed
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.testing.pytest import raises, XFAIL
def test_Pow_is_pos_neg():
    z = Symbol('z', real=True)
    w = Symbol('w', nonpositive=True)
    assert (S.NegativeOne ** S(2)).is_positive is True
    assert (S.One ** z).is_positive is True
    assert (S.NegativeOne ** S(3)).is_positive is False
    assert (S.Zero ** S.Zero).is_positive is True
    assert (w ** S(3)).is_positive is False
    assert (w ** S(2)).is_positive is None
    assert (I ** 2).is_positive is False
    assert (I ** 4).is_positive is True
    p = Symbol('p', zero=True)
    q = Symbol('q', zero=False, real=True)
    j = Symbol('j', zero=False, even=True)
    x = Symbol('x', zero=True)
    y = Symbol('y', zero=True)
    assert (p ** q).is_positive is False
    assert (p ** q).is_negative is False
    assert (p ** j).is_positive is False
    assert (x ** y).is_positive is True
    assert (x ** y).is_negative is False