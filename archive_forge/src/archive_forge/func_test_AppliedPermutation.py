from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_AppliedPermutation():
    x = Symbol('x')
    p = Permutation(0, 1, 2)
    raises(ValueError, lambda: AppliedPermutation((0, 1, 2), x))
    assert AppliedPermutation(p, 1, evaluate=True) == 2
    assert AppliedPermutation(p, 1, evaluate=False).__class__ == AppliedPermutation