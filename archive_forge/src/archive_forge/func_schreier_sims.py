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
def schreier_sims(self):
    """Schreier-Sims algorithm.

        Explanation
        ===========

        It computes the generators of the chain of stabilizers
        `G > G_{b_1} > .. > G_{b1,..,b_r} > 1`
        in which `G_{b_1,..,b_i}` stabilizes `b_1,..,b_i`,
        and the corresponding ``s`` cosets.
        An element of the group can be written as the product
        `h_1*..*h_s`.

        We use the incremental Schreier-Sims algorithm.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.schreier_sims()
        >>> G.basic_transversals
        [{0: (2)(0 1), 1: (2), 2: (1 2)},
         {0: (2), 2: (0 2)}]
        """
    if self._transversals:
        return
    self._schreier_sims()
    return