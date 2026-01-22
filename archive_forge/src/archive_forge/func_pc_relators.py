from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group
def pc_relators(self):
    """
        Return the polycyclic presentation.

        Explanation
        ===========

        There are two types of relations used in polycyclic
        presentation.

        * Power relations : Power relators are of the form `x_i^{re_i}`,
          where `i \\in \\{0, \\ldots, \\mathrm{len(pcgs)}\\}`, ``x`` represents polycyclic
          generator and ``re`` is the corresponding relative order.

        * Conjugate relations : Conjugate relators are of the form `x_j^-1x_ix_j`,
          where `j < i \\in \\{0, \\ldots, \\mathrm{len(pcgs)}\\}`.

        Returns
        =======

        A dictionary with power and conjugate relations as key and
        their collected form as corresponding values.

        Notes
        =====

        Identity Permutation is mapped with empty ``()``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> S = SymmetricGroup(49).sylow_subgroup(7)
        >>> der = S.derived_series()
        >>> G = der[len(der)-2]
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> len(pcgs)
        6
        >>> free_group = collector.free_group
        >>> pc_resentation = collector.pc_presentation
        >>> free_to_perm = {}
        >>> for s, g in zip(free_group.symbols, pcgs):
        ...     free_to_perm[s] = g

        >>> for k, v in pc_resentation.items():
        ...     k_array = k.array_form
        ...     if v != ():
        ...        v_array = v.array_form
        ...     lhs = Permutation()
        ...     for gen in k_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         lhs = lhs*free_to_perm[s]**e
        ...     if v == ():
        ...         assert lhs.is_identity
        ...         continue
        ...     rhs = Permutation()
        ...     for gen in v_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         rhs = rhs*free_to_perm[s]**e
        ...     assert lhs == rhs

        """
    free_group = self.free_group
    rel_order = self.relative_order
    pc_relators = {}
    perm_to_free = {}
    pcgs = self.pcgs
    for gen, s in zip(pcgs, free_group.generators):
        perm_to_free[gen ** (-1)] = s ** (-1)
        perm_to_free[gen] = s
    pcgs = pcgs[::-1]
    series = self.pc_series[::-1]
    rel_order = rel_order[::-1]
    collected_gens = []
    for i, gen in enumerate(pcgs):
        re = rel_order[i]
        relation = perm_to_free[gen] ** re
        G = series[i]
        l = G.generator_product(gen ** re, original=True)
        l.reverse()
        word = free_group.identity
        for g in l:
            word = word * perm_to_free[g]
        word = self.collected_word(word)
        pc_relators[relation] = word if word else ()
        self.pc_presentation = pc_relators
        collected_gens.append(gen)
        if len(collected_gens) > 1:
            conj = collected_gens[len(collected_gens) - 1]
            conjugator = perm_to_free[conj]
            for j in range(len(collected_gens) - 1):
                conjugated = perm_to_free[collected_gens[j]]
                relation = conjugator ** (-1) * conjugated * conjugator
                gens = conj ** (-1) * collected_gens[j] * conj
                l = G.generator_product(gens, original=True)
                l.reverse()
                word = free_group.identity
                for g in l:
                    word = word * perm_to_free[g]
                word = self.collected_word(word)
                pc_relators[relation] = word if word else ()
                self.pc_presentation = pc_relators
    return pc_relators