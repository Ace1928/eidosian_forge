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
def test_Mul_is_prime_composite():
    x = Symbol('x', positive=True, integer=True)
    y = Symbol('y', positive=True, integer=True)
    assert (x * y).is_prime is None
    assert ((x + 1) * (y + 1)).is_prime is False
    assert ((x + 1) * (y + 1)).is_composite is True
    x = Symbol('x', positive=True)
    assert ((x + 1) * (y + 1)).is_prime is None
    assert ((x + 1) * (y + 1)).is_composite is None