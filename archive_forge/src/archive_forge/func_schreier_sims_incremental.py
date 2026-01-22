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
def schreier_sims_incremental(self, base=None, gens=None, slp_dict=False):
    """Extend a sequence of points and generating set to a base and strong
        generating set.

        Parameters
        ==========

        base
            The sequence of points to be extended to a base. Optional
            parameter with default value ``[]``.
        gens
            The generating set to be extended to a strong generating set
            relative to the base obtained. Optional parameter with default
            value ``self.generators``.

        slp_dict
            If `True`, return a dictionary `{g: gens}` for each strong
            generator `g` where `gens` is a list of strong generators
            coming before `g` in `strong_gens`, such that the product
            of the elements of `gens` is equal to `g`.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the base obtained, and ``strong_gens`` is the strong
            generating set relative to it. The original parameters ``base``,
            ``gens`` remain unchanged.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> A = AlternatingGroup(7)
        >>> base = [2, 3]
        >>> seq = [2, 3]
        >>> base, strong_gens = A.schreier_sims_incremental(base=seq)
        >>> _verify_bsgs(A, base, strong_gens)
        True
        >>> base[:2]
        [2, 3]

        Notes
        =====

        This version of the Schreier-Sims algorithm runs in polynomial time.
        There are certain assumptions in the implementation - if the trivial
        group is provided, ``base`` and ``gens`` are returned immediately,
        as any sequence of points is a base for the trivial group. If the
        identity is present in the generators ``gens``, it is removed as
        it is a redundant generator.
        The implementation is described in [1], pp. 90-93.

        See Also
        ========

        schreier_sims, schreier_sims_random

        """
    if base is None:
        base = []
    if gens is None:
        gens = self.generators[:]
    degree = self.degree
    id_af = list(range(degree))
    if len(gens) == 1 and gens[0].is_Identity:
        if slp_dict:
            return (base, gens, {gens[0]: [gens[0]]})
        return (base, gens)
    _base, _gens = (base[:], gens[:])
    _gens = [x for x in _gens if not x.is_Identity]
    for gen in _gens:
        if all((x == gen._array_form[x] for x in _base)):
            for new in id_af:
                if gen._array_form[new] != new:
                    break
            else:
                assert None
            _base.append(new)
    strong_gens_distr = _distribute_gens_by_base(_base, _gens)
    strong_gens_slp = []
    orbs = {}
    transversals = {}
    slps = {}
    base_len = len(_base)
    for i in range(base_len):
        transversals[i], slps[i] = _orbit_transversal(degree, strong_gens_distr[i], _base[i], pairs=True, af=True, slp=True)
        transversals[i] = dict(transversals[i])
        orbs[i] = list(transversals[i].keys())
    i = base_len - 1
    while i >= 0:
        continue_i = False
        db = {}
        for beta, u_beta in list(transversals[i].items()):
            for j, gen in enumerate(strong_gens_distr[i]):
                gb = gen._array_form[beta]
                u1 = transversals[i][gb]
                g1 = _af_rmul(gen._array_form, u_beta)
                slp = [(i, g) for g in slps[i][beta]]
                slp = [(i, j)] + slp
                if g1 != u1:
                    y = True
                    try:
                        u1_inv = db[gb]
                    except KeyError:
                        u1_inv = db[gb] = _af_invert(u1)
                    schreier_gen = _af_rmul(u1_inv, g1)
                    u1_inv_slp = slps[i][gb][:]
                    u1_inv_slp.reverse()
                    u1_inv_slp = [(i, (g,)) for g in u1_inv_slp]
                    slp = u1_inv_slp + slp
                    h, j, slp = _strip_af(schreier_gen, _base, orbs, transversals, i, slp=slp, slps=slps)
                    if j <= base_len:
                        y = False
                    elif h:
                        y = False
                        moved = 0
                        while h[moved] == moved:
                            moved += 1
                        _base.append(moved)
                        base_len += 1
                        strong_gens_distr.append([])
                    if y is False:
                        h = _af_new(h)
                        strong_gens_slp.append((h, slp))
                        for l in range(i + 1, j):
                            strong_gens_distr[l].append(h)
                            transversals[l], slps[l] = _orbit_transversal(degree, strong_gens_distr[l], _base[l], pairs=True, af=True, slp=True)
                            transversals[l] = dict(transversals[l])
                            orbs[l] = list(transversals[l].keys())
                        i = j - 1
                        continue_i = True
                if continue_i is True:
                    break
            if continue_i is True:
                break
        if continue_i is True:
            continue
        i -= 1
    strong_gens = _gens[:]
    if slp_dict:
        for k, slp in strong_gens_slp:
            strong_gens.append(k)
            for i in range(len(slp)):
                s = slp[i]
                if isinstance(s[1], tuple):
                    slp[i] = strong_gens_distr[s[0]][s[1][0]] ** (-1)
                else:
                    slp[i] = strong_gens_distr[s[0]][s[1]]
        strong_gens_slp = dict(strong_gens_slp)
        for g in _gens:
            strong_gens_slp[g] = [g]
        return (_base, strong_gens, strong_gens_slp)
    strong_gens.extend([k for k, _ in strong_gens_slp])
    return (_base, strong_gens)