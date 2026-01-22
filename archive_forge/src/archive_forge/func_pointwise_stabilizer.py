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
def pointwise_stabilizer(self, points, incremental=True):
    """Return the pointwise stabilizer for a set of points.

        Explanation
        ===========

        For a permutation group `G` and a set of points
        `\\{p_1, p_2,\\ldots, p_k\\}`, the pointwise stabilizer of
        `p_1, p_2, \\ldots, p_k` is defined as
        `G_{p_1,\\ldots, p_k} =
        \\{g\\in G | g(p_i) = p_i \\forall i\\in\\{1, 2,\\ldots,k\\}\\}` ([1],p20).
        It is a subgroup of `G`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(7)
        >>> Stab = S.pointwise_stabilizer([2, 3, 5])
        >>> Stab.is_subgroup(S.stabilizer(2).stabilizer(3).stabilizer(5))
        True

        See Also
        ========

        stabilizer, schreier_sims_incremental

        Notes
        =====

        When incremental == True,
        rather than the obvious implementation using successive calls to
        ``.stabilizer()``, this uses the incremental Schreier-Sims algorithm
        to obtain a base with starting segment - the given points.

        """
    if incremental:
        base, strong_gens = self.schreier_sims_incremental(base=points)
        stab_gens = []
        degree = self.degree
        for gen in strong_gens:
            if [gen(point) for point in points] == points:
                stab_gens.append(gen)
        if not stab_gens:
            stab_gens = _af_new(list(range(degree)))
        return PermutationGroup(stab_gens)
    else:
        gens = self._generators
        degree = self.degree
        for x in points:
            gens = _stabilizer(degree, gens, x)
    return PermutationGroup(gens)