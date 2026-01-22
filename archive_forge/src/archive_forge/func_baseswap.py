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
def baseswap(self, base, strong_gens, pos, randomized=False, transversals=None, basic_orbits=None, strong_gens_distr=None):
    """Swap two consecutive base points in base and strong generating set.

        Explanation
        ===========

        If a base for a group `G` is given by `(b_1, b_2, \\dots, b_k)`, this
        function returns a base `(b_1, b_2, \\dots, b_{i+1}, b_i, \\dots, b_k)`,
        where `i` is given by ``pos``, and a strong generating set relative
        to that base. The original base and strong generating set are not
        modified.

        The randomized version (default) is of Las Vegas type.

        Parameters
        ==========

        base, strong_gens
            The base and strong generating set.
        pos
            The position at which swapping is performed.
        randomized
            A switch between randomized and deterministic version.
        transversals
            The transversals for the basic orbits, if known.
        basic_orbits
            The basic orbits, if known.
        strong_gens_distr
            The strong generators distributed by basic stabilizers, if known.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the new base, and ``strong_gens`` is a generating set
            relative to it.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> S = SymmetricGroup(4)
        >>> S.schreier_sims()
        >>> S.base
        [0, 1, 2]
        >>> base, gens = S.baseswap(S.base, S.strong_gens, 1, randomized=False)
        >>> base, gens
        ([0, 2, 1],
        [(0 1 2 3), (3)(0 1), (1 3 2),
         (2 3), (1 3)])

        check that base, gens is a BSGS

        >>> S1 = PermutationGroup(gens)
        >>> _verify_bsgs(S1, base, gens)
        True

        See Also
        ========

        schreier_sims

        Notes
        =====

        The deterministic version of the algorithm is discussed in
        [1], pp. 102-103; the randomized version is discussed in [1], p.103, and
        [2], p.98. It is of Las Vegas type.
        Notice that [1] contains a mistake in the pseudocode and
        discussion of BASESWAP: on line 3 of the pseudocode,
        `|\\beta_{i+1}^{\\left\\langle T\\right\\rangle}|` should be replaced by
        `|\\beta_{i}^{\\left\\langle T\\right\\rangle}|`, and the same for the
        discussion of the algorithm.

        """
    transversals, basic_orbits, strong_gens_distr = _handle_precomputed_bsgs(base, strong_gens, transversals, basic_orbits, strong_gens_distr)
    base_len = len(base)
    degree = self.degree
    size = len(basic_orbits[pos]) * len(basic_orbits[pos + 1]) // len(_orbit(degree, strong_gens_distr[pos], base[pos + 1]))
    if pos + 2 > base_len - 1:
        T = []
    else:
        T = strong_gens_distr[pos + 2][:]
    if randomized is True:
        stab_pos = PermutationGroup(strong_gens_distr[pos])
        schreier_vector = stab_pos.schreier_vector(base[pos + 1])
        while len(_orbit(degree, T, base[pos])) != size:
            new = stab_pos.random_stab(base[pos + 1], schreier_vector=schreier_vector)
            T.append(new)
    else:
        Gamma = set(basic_orbits[pos])
        Gamma.remove(base[pos])
        if base[pos + 1] in Gamma:
            Gamma.remove(base[pos + 1])
        while len(_orbit(degree, T, base[pos])) != size:
            gamma = next(iter(Gamma))
            x = transversals[pos][gamma]
            temp = x._array_form.index(base[pos + 1])
            if temp not in basic_orbits[pos + 1]:
                Gamma = Gamma - _orbit(degree, T, gamma)
            else:
                y = transversals[pos + 1][temp]
                el = rmul(x, y)
                if el(base[pos]) not in _orbit(degree, T, base[pos]):
                    T.append(el)
                    Gamma = Gamma - _orbit(degree, T, base[pos])
    strong_gens_new_distr = strong_gens_distr[:]
    strong_gens_new_distr[pos + 1] = T
    base_new = base[:]
    base_new[pos], base_new[pos + 1] = (base_new[pos + 1], base_new[pos])
    strong_gens_new = _strong_gens_from_distr(strong_gens_new_distr)
    for gen in T:
        if gen not in strong_gens_new:
            strong_gens_new.append(gen)
    return (base_new, strong_gens_new)