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
def test_pre_generated_assumption_rules_are_valid():
    pre_generated_assumptions = _load_pre_generated_assumption_rules()
    generated_assumptions = _generate_assumption_rules()
    assert pre_generated_assumptions._to_python() == generated_assumptions._to_python(), 'pre-generated assumptions are invalid, see sympy.core.assumptions._generate_assumption_rules'