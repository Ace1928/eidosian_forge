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
def test_assumptions_copy():
    assert assumptions(Symbol('x'), {'commutative': True}) == {'commutative': True}
    assert assumptions(Symbol('x'), ['integer']) == {}
    assert assumptions(Symbol('x'), ['commutative']) == {'commutative': True}
    assert assumptions(Symbol('x')) == {'commutative': True}
    assert assumptions(1)['positive']
    assert assumptions(3 + I) == {'algebraic': True, 'commutative': True, 'complex': True, 'composite': False, 'even': False, 'extended_negative': False, 'extended_nonnegative': False, 'extended_nonpositive': False, 'extended_nonzero': False, 'extended_positive': False, 'extended_real': False, 'finite': True, 'imaginary': False, 'infinite': False, 'integer': False, 'irrational': False, 'negative': False, 'noninteger': False, 'nonnegative': False, 'nonpositive': False, 'nonzero': False, 'odd': False, 'positive': False, 'prime': False, 'rational': False, 'real': False, 'transcendental': False, 'zero': False}