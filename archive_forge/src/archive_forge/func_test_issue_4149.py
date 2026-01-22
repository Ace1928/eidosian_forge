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
def test_issue_4149():
    assert (3 + I).is_complex
    assert (3 + I).is_imaginary is False
    assert (3 * I + S.Pi * I).is_imaginary
    y = Symbol('y', real=True)
    assert (3 * I + S.Pi * I + y * I).is_imaginary is None
    p = Symbol('p', positive=True)
    assert (3 * I + S.Pi * I + p * I).is_imaginary
    n = Symbol('n', negative=True)
    assert (-3 * I - S.Pi * I + n * I).is_imaginary
    i = Symbol('i', imaginary=True)
    assert [(i ** a).is_imaginary for a in range(4)] == [False, True, False, True]
    e = S('-sqrt(3)*I/2 + 0.866025403784439*I')
    assert e.is_real is False
    assert e.is_imaginary