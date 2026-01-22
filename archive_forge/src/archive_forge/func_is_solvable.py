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
def is_solvable(self):
    """Test if the group is solvable.

        ``G`` is solvable if its derived series terminates with the trivial
        group ([1], p.29).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(3)
        >>> S.is_solvable
        True

        See Also
        ========

        is_nilpotent, derived_series

        """
    if self._is_solvable is None:
        if self.order() % 2 != 0:
            return True
        ds = self.derived_series()
        terminator = ds[len(ds) - 1]
        gens = terminator.generators
        degree = self.degree
        identity = _af_new(list(range(degree)))
        if all((g == identity for g in gens)):
            self._is_solvable = True
            return True
        else:
            self._is_solvable = False
            return False
    else:
        return self._is_solvable