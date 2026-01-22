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
def coset_factor(self, g, factor_index=False):
    """Return ``G``'s (self's) coset factorization of ``g``

        Explanation
        ===========

        If ``g`` is an element of ``G`` then it can be written as the product
        of permutations drawn from the Schreier-Sims coset decomposition,

        The permutations returned in ``f`` are those for which
        the product gives ``g``: ``g = f[n]*...f[1]*f[0]`` where ``n = len(B)``
        and ``B = G.base``. f[i] is one of the permutations in
        ``self._basic_orbits[i]``.

        If factor_index==True,
        returns a tuple ``[b[0],..,b[n]]``, where ``b[i]``
        belongs to ``self._basic_orbits[i]``

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])

        Define g:

        >>> g = Permutation(7)(1, 2, 4)(3, 6, 5)

        Confirm that it is an element of G:

        >>> G.contains(g)
        True

        Thus, it can be written as a product of factors (up to
        3) drawn from u. See below that a factor from u1 and u2
        and the Identity permutation have been used:

        >>> f = G.coset_factor(g)
        >>> f[2]*f[1]*f[0] == g
        True
        >>> f1 = G.coset_factor(g, True); f1
        [0, 4, 4]
        >>> tr = G.basic_transversals
        >>> f[0] == tr[0][f1[0]]
        True

        If g is not an element of G then [] is returned:

        >>> c = Permutation(5, 6, 7)
        >>> G.coset_factor(c)
        []

        See Also
        ========

        sympy.combinatorics.util._strip

        """
    if isinstance(g, (Cycle, Permutation)):
        g = g.list()
    if len(g) != self._degree:
        raise ValueError('g should be the same size as permutations of G')
    I = list(range(self._degree))
    basic_orbits = self.basic_orbits
    transversals = self._transversals
    factors = []
    base = self.base
    h = g
    for i in range(len(base)):
        beta = h[base[i]]
        if beta == base[i]:
            factors.append(beta)
            continue
        if beta not in basic_orbits[i]:
            return []
        u = transversals[i][beta]._array_form
        h = _af_rmul(_af_invert(u), h)
        factors.append(beta)
    if h != I:
        return []
    if factor_index:
        return factors
    tr = self.basic_transversals
    factors = [tr[i][factors[i]] for i in range(len(base))]
    return factors