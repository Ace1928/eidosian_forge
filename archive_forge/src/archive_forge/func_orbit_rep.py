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
def orbit_rep(self, alpha, beta, schreier_vector=None):
    """Return a group element which sends ``alpha`` to ``beta``.

        Explanation
        ===========

        If ``beta`` is not in the orbit of ``alpha``, the function returns
        ``False``. This implementation makes use of the schreier vector.
        For a proof of correctness, see [1], p.80

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> G = AlternatingGroup(5)
        >>> G.orbit_rep(0, 4)
        (0 4 1 2 3)

        See Also
        ========

        schreier_vector

        """
    if schreier_vector is None:
        schreier_vector = self.schreier_vector(alpha)
    if schreier_vector[beta] is None:
        return False
    k = schreier_vector[beta]
    gens = [x._array_form for x in self.generators]
    a = []
    while k != -1:
        a.append(gens[k])
        beta = gens[k].index(beta)
        k = schreier_vector[beta]
    if a:
        return _af_new(_af_rmuln(*a))
    else:
        return _af_new(list(range(self._degree)))