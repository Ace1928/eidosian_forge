from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
@property
def is_perfect(self):
    """Return ``True`` if the group is perfect.
        A group is perfect if it equals to its derived subgroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(1,2,3)(4,5)
        >>> b = Permutation(1,2,3,4,5)
        >>> G = PermutationGroup([a, b])
        >>> G.is_perfect
        False

        """
    if self._is_perfect is None:
        self._is_perfect = self.equals(self.derived_subgroup())
    return self._is_perfect