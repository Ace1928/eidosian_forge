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
def is_primitive(self, randomized=True):
    """Test if a group is primitive.

        Explanation
        ===========

        A permutation group ``G`` acting on a set ``S`` is called primitive if
        ``S`` contains no nontrivial block under the action of ``G``
        (a block is nontrivial if its cardinality is more than ``1``).

        Notes
        =====

        The algorithm is described in [1], p.83, and uses the function
        minimal_block to search for blocks of the form `\\{0, k\\}` for ``k``
        ranging over representatives for the orbits of `G_0`, the stabilizer of
        ``0``. This algorithm has complexity `O(n^2)` where ``n`` is the degree
        of the group, and will perform badly if `G_0` is small.

        There are two implementations offered: one finds `G_0`
        deterministically using the function ``stabilizer``, and the other
        (default) produces random elements of `G_0` using ``random_stab``,
        hoping that they generate a subgroup of `G_0` with not too many more
        orbits than `G_0` (this is suggested in [1], p.83). Behavior is changed
        by the ``randomized`` flag.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_primitive()
        False

        See Also
        ========

        minimal_block, random_stab

        """
    if self._is_primitive is not None:
        return self._is_primitive
    if self.is_transitive() is False:
        return False
    if randomized:
        random_stab_gens = []
        v = self.schreier_vector(0)
        for _ in range(len(self)):
            random_stab_gens.append(self.random_stab(0, v))
        stab = PermutationGroup(random_stab_gens)
    else:
        stab = self.stabilizer(0)
    orbits = stab.orbits()
    for orb in orbits:
        x = orb.pop()
        if x != 0 and any((e != 0 for e in self.minimal_block([0, x]))):
            self._is_primitive = False
            return False
    self._is_primitive = True
    return True