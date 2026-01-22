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
def coset_unrank(self, rank, af=False):
    """unrank using Schreier-Sims representation

        coset_unrank is the inverse operation of coset_rank
        if 0 <= rank < order; otherwise it returns None.

        """
    if rank < 0 or rank >= self.order():
        return None
    base = self.base
    transversals = self.basic_transversals
    basic_orbits = self.basic_orbits
    m = len(base)
    v = [0] * m
    for i in range(m):
        rank, c = divmod(rank, len(transversals[i]))
        v[i] = basic_orbits[i][c]
    a = [transversals[i][v[i]]._array_form for i in range(m)]
    h = _af_rmuln(*a)
    if af:
        return h
    else:
        return _af_new(h)