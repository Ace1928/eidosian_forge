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
def schreier_sims_random(self, base=None, gens=None, consec_succ=10, _random_prec=None):
    """Randomized Schreier-Sims algorithm.

        Explanation
        ===========

        The randomized Schreier-Sims algorithm takes the sequence ``base``
        and the generating set ``gens``, and extends ``base`` to a base, and
        ``gens`` to a strong generating set relative to that base with
        probability of a wrong answer at most `2^{-consec\\_succ}`,
        provided the random generators are sufficiently random.

        Parameters
        ==========

        base
            The sequence to be extended to a base.
        gens
            The generating set to be extended to a strong generating set.
        consec_succ
            The parameter defining the probability of a wrong answer.
        _random_prec
            An internal parameter used for testing purposes.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the base and ``strong_gens`` is the strong generating
            set relative to it.

        Examples
        ========

        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(5)
        >>> base, strong_gens = S.schreier_sims_random(consec_succ=5)
        >>> _verify_bsgs(S, base, strong_gens) #doctest: +SKIP
        True

        Notes
        =====

        The algorithm is described in detail in [1], pp. 97-98. It extends
        the orbits ``orbs`` and the permutation groups ``stabs`` to
        basic orbits and basic stabilizers for the base and strong generating
        set produced in the end.
        The idea of the extension process
        is to "sift" random group elements through the stabilizer chain
        and amend the stabilizers/orbits along the way when a sift
        is not successful.
        The helper function ``_strip`` is used to attempt
        to decompose a random group element according to the current
        state of the stabilizer chain and report whether the element was
        fully decomposed (successful sift) or not (unsuccessful sift). In
        the latter case, the level at which the sift failed is reported and
        used to amend ``stabs``, ``base``, ``gens`` and ``orbs`` accordingly.
        The halting condition is for ``consec_succ`` consecutive successful
        sifts to pass. This makes sure that the current ``base`` and ``gens``
        form a BSGS with probability at least `1 - 1/\\text{consec\\_succ}`.

        See Also
        ========

        schreier_sims

        """
    if base is None:
        base = []
    if gens is None:
        gens = self.generators
    base_len = len(base)
    n = self.degree
    for gen in gens:
        if all((gen(x) == x for x in base)):
            new = 0
            while gen._array_form[new] == new:
                new += 1
            base.append(new)
            base_len += 1
    strong_gens_distr = _distribute_gens_by_base(base, gens)
    transversals = {}
    orbs = {}
    for i in range(base_len):
        transversals[i] = dict(_orbit_transversal(n, strong_gens_distr[i], base[i], pairs=True))
        orbs[i] = list(transversals[i].keys())
    c = 0
    while c < consec_succ:
        if _random_prec is None:
            g = self.random_pr()
        else:
            g = _random_prec['g'].pop()
        h, j = _strip(g, base, orbs, transversals)
        y = True
        if j <= base_len:
            y = False
        elif not h.is_Identity:
            y = False
            moved = 0
            while h(moved) == moved:
                moved += 1
            base.append(moved)
            base_len += 1
            strong_gens_distr.append([])
        if y is False:
            for l in range(1, j):
                strong_gens_distr[l].append(h)
                transversals[l] = dict(_orbit_transversal(n, strong_gens_distr[l], base[l], pairs=True))
                orbs[l] = list(transversals[l].keys())
            c = 0
        else:
            c += 1
    strong_gens = strong_gens_distr[0][:]
    for gen in strong_gens_distr[1]:
        if gen not in strong_gens:
            strong_gens.append(gen)
    return (base, strong_gens)