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
def polycyclic_group(self):
    """
        Return the PolycyclicGroup instance with below parameters:

        Explanation
        ===========

        * pc_sequence : Polycyclic sequence is formed by collecting all
          the missing generators between the adjacent groups in the
          derived series of given permutation group.

        * pc_series : Polycyclic series is formed by adding all the missing
          generators of ``der[i+1]`` in ``der[i]``, where ``der`` represents
          the derived series.

        * relative_order : A list, computed by the ratio of adjacent groups in
          pc_series.

        """
    from sympy.combinatorics.pc_groups import PolycyclicGroup
    if not self.is_polycyclic:
        raise ValueError('The group must be solvable')
    der = self.derived_series()
    pc_series = []
    pc_sequence = []
    relative_order = []
    pc_series.append(der[-1])
    der.reverse()
    for i in range(len(der) - 1):
        H = der[i]
        for g in der[i + 1].generators:
            if g not in H:
                H = PermutationGroup([g] + H.generators)
                pc_series.insert(0, H)
                pc_sequence.insert(0, g)
                G1 = pc_series[0].order()
                G2 = pc_series[1].order()
                relative_order.insert(0, G1 // G2)
    return PolycyclicGroup(pc_sequence, pc_series, relative_order, collector=None)