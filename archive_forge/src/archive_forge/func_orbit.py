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
def orbit(self, alpha, action='tuples'):
    """Compute the orbit of alpha `\\{g(\\alpha) | g \\in G\\}` as a set.

        Explanation
        ===========

        The time complexity of the algorithm used here is `O(|Orb|*r)` where
        `|Orb|` is the size of the orbit and ``r`` is the number of generators of
        the group. For a more detailed analysis, see [1], p.78, [2], pp. 19-21.
        Here alpha can be a single point, or a list of points.

        If alpha is a single point, the ordinary orbit is computed.
        if alpha is a list of points, there are three available options:

        'union' - computes the union of the orbits of the points in the list
        'tuples' - computes the orbit of the list interpreted as an ordered
        tuple under the group action ( i.e., g((1,2,3)) = (g(1), g(2), g(3)) )
        'sets' - computes the orbit of the list interpreted as a sets

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 2, 0, 4, 5, 6, 3])
        >>> G = PermutationGroup([a])
        >>> G.orbit(0)
        {0, 1, 2}
        >>> G.orbit([0, 4], 'union')
        {0, 1, 2, 3, 4, 5, 6}

        See Also
        ========

        orbit_transversal

        """
    return _orbit(self.degree, self.generators, alpha, action)