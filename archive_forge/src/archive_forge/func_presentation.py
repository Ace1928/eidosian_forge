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
def presentation(self, eliminate_gens=True):
    """
        Return an `FpGroup` presentation of the group.

        The algorithm is described in [1], Chapter 6.1.

        """
    from sympy.combinatorics.fp_groups import FpGroup, simplify_presentation
    from sympy.combinatorics.coset_table import CosetTable
    from sympy.combinatorics.free_groups import free_group
    from sympy.combinatorics.homomorphisms import homomorphism
    if self._fp_presentation:
        return self._fp_presentation

    def _factor_group_by_rels(G, rels):
        if isinstance(G, FpGroup):
            rels.extend(G.relators)
            return FpGroup(G.free_group, list(set(rels)))
        return FpGroup(G, rels)
    gens = self.generators
    len_g = len(gens)
    if len_g == 1:
        order = gens[0].order()
        if order == 1:
            return free_group([])[0]
        F, x = free_group('x')
        return FpGroup(F, [x ** order])
    if self.order() > 20:
        half_gens = self.generators[0:(len_g + 1) // 2]
    else:
        half_gens = []
    H = PermutationGroup(half_gens)
    H_p = H.presentation()
    len_h = len(H_p.generators)
    C = self.coset_table(H)
    n = len(C)
    gen_syms = ['x_%d' % i for i in range(len(gens))]
    F = free_group(', '.join(gen_syms))[0]
    images = [F.generators[i] for i in range(len_h)]
    R = homomorphism(H_p, F, H_p.generators, images, check=False)
    rels = R(H_p.relators)
    G_p = FpGroup(F, rels)
    T = homomorphism(G_p, self, G_p.generators, gens)
    C_p = CosetTable(G_p, [])
    C_p.table = [[None] * (2 * len_g) for i in range(n)]
    transversal = [None] * n
    transversal[0] = G_p.identity
    for i in range(2 * len_h):
        C_p.table[0][i] = 0
    gamma = 1
    for alpha, x in product(range(n), range(2 * len_g)):
        beta = C[alpha][x]
        if beta == gamma:
            gen = G_p.generators[x // 2] ** (-1) ** (x % 2)
            transversal[beta] = transversal[alpha] * gen
            C_p.table[alpha][x] = beta
            C_p.table[beta][x + (-1) ** (x % 2)] = alpha
            gamma += 1
            if gamma == n:
                break
    C_p.p = list(range(n))
    beta = x = 0
    while not C_p.is_complete():
        while C_p.table[beta][x] == C[beta][x]:
            x = (x + 1) % (2 * len_g)
            if x == 0:
                beta = (beta + 1) % n
        gen = G_p.generators[x // 2] ** (-1) ** (x % 2)
        new_rel = transversal[beta] * gen * transversal[C[beta][x]] ** (-1)
        perm = T(new_rel)
        nxt = G_p.identity
        for s in H.generator_product(perm, original=True):
            nxt = nxt * T.invert(s) ** (-1)
        new_rel = new_rel * nxt
        G_p = _factor_group_by_rels(G_p, [new_rel])
        C_p.scan_and_fill(0, new_rel)
        C_p = G_p.coset_enumeration([], strategy='coset_table', draft=C_p, max_cosets=n, incomplete=True)
    self._fp_presentation = simplify_presentation(G_p)
    return self._fp_presentation