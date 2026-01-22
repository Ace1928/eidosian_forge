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
def commutator(self, G, H):
    """
        Return the commutator of two subgroups.

        Explanation
        ===========

        For a permutation group ``K`` and subgroups ``G``, ``H``, the
        commutator of ``G`` and ``H`` is defined as the group generated
        by all the commutators `[g, h] = hgh^{-1}g^{-1}` for ``g`` in ``G`` and
        ``h`` in ``H``. It is naturally a subgroup of ``K`` ([1], p.27).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> S = SymmetricGroup(5)
        >>> A = AlternatingGroup(5)
        >>> G = S.commutator(S, A)
        >>> G.is_subgroup(A)
        True

        See Also
        ========

        derived_subgroup

        Notes
        =====

        The commutator of two subgroups `H, G` is equal to the normal closure
        of the commutators of all the generators, i.e. `hgh^{-1}g^{-1}` for `h`
        a generator of `H` and `g` a generator of `G` ([1], p.28)

        """
    ggens = G.generators
    hgens = H.generators
    commutators = []
    for ggen in ggens:
        for hgen in hgens:
            commutator = rmul(hgen, ggen, ~hgen, ~ggen)
            if commutator not in commutators:
                commutators.append(commutator)
    res = self.normal_closure(commutators)
    return res