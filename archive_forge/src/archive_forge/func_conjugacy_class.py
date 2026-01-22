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
def conjugacy_class(self, x):
    """Return the conjugacy class of an element in the group.

        Explanation
        ===========

        The conjugacy class of an element ``g`` in a group ``G`` is the set of
        elements ``x`` in ``G`` that are conjugate with ``g``, i.e. for which

            ``g = xax^{-1}``

        for some ``a`` in ``G``.

        Note that conjugacy is an equivalence relation, and therefore that
        conjugacy classes are partitions of ``G``. For a list of all the
        conjugacy classes of the group, use the conjugacy_classes() method.

        In a permutation group, each conjugacy class corresponds to a particular
        `cycle structure': for example, in ``S_3``, the conjugacy classes are:

            * the identity class, ``{()}``
            * all transpositions, ``{(1 2), (1 3), (2 3)}``
            * all 3-cycles, ``{(1 2 3), (1 3 2)}``

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, SymmetricGroup
        >>> S3 = SymmetricGroup(3)
        >>> S3.conjugacy_class(Permutation(0, 1, 2))
        {(0 1 2), (0 2 1)}

        Notes
        =====

        This procedure computes the conjugacy class directly by finding the
        orbit of the element under conjugation in G. This algorithm is only
        feasible for permutation groups of relatively small order, but is like
        the orbit() function itself in that respect.
        """
    new_class = {x}
    last_iteration = new_class
    while len(last_iteration) > 0:
        this_iteration = set()
        for y in last_iteration:
            for s in self.generators:
                conjugated = s * y * ~s
                if conjugated not in new_class:
                    this_iteration.add(conjugated)
        new_class.update(last_iteration)
        last_iteration = this_iteration
    return new_class