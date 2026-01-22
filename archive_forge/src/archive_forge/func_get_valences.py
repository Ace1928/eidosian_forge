from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_valences(self, structure: Structure):
    """
        Returns a list of valences for each site in the structure.

        Args:
            structure: Structure to analyze

        Returns:
            A list of valences for each site in the structure (for an ordered structure),
            e.g., [1, 1, -2] or a list of lists with the valences for each fractional
            element of each site in the structure (for an unordered structure), e.g., [[2,
            4], [3], [-2], [-2], [-2]]

        Raises:
            A ValueError if the valences cannot be determined.
        """
    els = [Element(el.symbol) for el in structure.elements]
    if (diff := (set(els) - set(BV_PARAMS))):
        raise ValueError(f'Structure contains elements not in set of BV parameters: {diff}')
    if self.symm_tol:
        finder = SpacegroupAnalyzer(structure, self.symm_tol)
        symm_structure = finder.get_symmetrized_structure()
        equi_sites = symm_structure.equivalent_sites
    else:
        equi_sites = [[site] for site in structure]
    equi_sites = sorted(equi_sites, key=lambda sites: -sites[0].species.average_electroneg)
    valences = []
    all_prob = []
    if structure.is_ordered:
        for sites in equi_sites:
            test_site = sites[0]
            nn = structure.get_neighbors(test_site, self.max_radius)
            prob = self._calc_site_probabilities(test_site, nn)
            all_prob.append(prob)
            val = list(prob)
            val = sorted(val, key=lambda v: -prob[v])
            valences.append(list(filter(lambda v: prob[v] > 0.01 * prob[val[0]], val)))
    else:
        full_all_prob = []
        for sites in equi_sites:
            test_site = sites[0]
            nn = structure.get_neighbors(test_site, self.max_radius)
            prob = self._calc_site_probabilities_unordered(test_site, nn)
            all_prob.append(prob)
            full_all_prob.extend(prob.values())
            vals = []
            for elem, _ in get_z_ordered_elmap(test_site.species):
                val = list(prob[elem.symbol])
                val = sorted(val, key=lambda v: -prob[elem.symbol][v])
                filtered = list(filter(lambda v: prob[elem.symbol][v] > 0.001 * prob[elem.symbol][val[0]], val))
                vals.append(filtered)
            valences.append(vals)
    if structure.is_ordered:
        n_sites = np.array(list(map(len, equi_sites)))
        valence_min = np.array(list(map(min, valences)))
        valence_max = np.array(list(map(max, valences)))
        self._n = 0
        self._best_score = 0
        self._best_vset = None

        def evaluate_assignment(v_set):
            el_oxi = collections.defaultdict(list)
            for idx, sites in enumerate(equi_sites):
                el_oxi[sites[0].specie.symbol].append(v_set[idx])
            max_diff = max((max(v) - min(v) for v in el_oxi.values()))
            if max_diff > 1:
                return
            score = functools.reduce(operator.mul, [all_prob[idx][val] for idx, val in enumerate(v_set)])
            if score > self._best_score:
                self._best_vset = v_set
                self._best_score = score

        def _recurse(assigned=None):
            if self._n > self.max_permutations:
                return
            if assigned is None:
                assigned = []
            i = len(assigned)
            highest = valence_max.copy()
            highest[:i] = assigned
            highest *= n_sites
            highest = np.sum(highest)
            lowest = valence_min.copy()
            lowest[:i] = assigned
            lowest *= n_sites
            lowest = np.sum(lowest)
            if highest < 0 or lowest > 0:
                self._n += 1
                return
            if i == len(valences):
                evaluate_assignment(assigned)
                self._n += 1
                return
            for v in valences[i]:
                new_assigned = list(assigned)
                _recurse([*new_assigned, v])
            return
    else:
        n_sites = np.array([len(sites) for sites in equi_sites])
        tmp = []
        attrib = []
        for idx, n_site in enumerate(n_sites):
            for _ in valences[idx]:
                tmp.append(n_site)
                attrib.append(idx)
        new_n_sites = np.array(tmp)
        fractions = []
        elements = []
        for sites in equi_sites:
            for sp, occu in get_z_ordered_elmap(sites[0].species):
                elements.append(sp.symbol)
                fractions.append(occu)
        fractions = np.array(fractions, float)
        new_valences = [val for vals in valences for val in vals]
        valence_min = np.array([min(val) for val in new_valences], float)
        valence_max = np.array([max(val) for val in new_valences], float)
        self._n = 0
        self._best_score = 0
        self._best_vset = None

        def evaluate_assignment(v_set):
            el_oxi = collections.defaultdict(list)
            jj = 0
            for sites in equi_sites:
                for specie, _ in get_z_ordered_elmap(sites[0].species):
                    el_oxi[specie.symbol].append(v_set[jj])
                    jj += 1
            max_diff = max((max(v) - min(v) for v in el_oxi.values()))
            if max_diff > 2:
                return
            score = functools.reduce(operator.mul, [all_prob[attrib[iv]][elements[iv]][vv] for iv, vv in enumerate(v_set)])
            if score > self._best_score:
                self._best_vset = v_set
                self._best_score = score

        def _recurse(assigned=None):
            if self._n > self.max_permutations:
                return
            if assigned is None:
                assigned = []
            i = len(assigned)
            highest = valence_max.copy()
            highest[:i] = assigned
            highest *= new_n_sites
            highest *= fractions
            highest = np.sum(highest)
            lowest = valence_min.copy()
            lowest[:i] = assigned
            lowest *= new_n_sites
            lowest *= fractions
            lowest = np.sum(lowest)
            if highest < -self.charge_neutrality_tolerance or lowest > self.charge_neutrality_tolerance:
                self._n += 1
                return
            if i == len(new_valences):
                evaluate_assignment(assigned)
                self._n += 1
                return
            for v in new_valences[i]:
                new_assigned = list(assigned)
                _recurse([*new_assigned, v])
            return
    _recurse()
    if self._best_vset:
        if structure.is_ordered:
            assigned = {}
            for val, sites in zip(self._best_vset, equi_sites):
                for site in sites:
                    assigned[site] = val
            return [int(assigned[site]) for site in structure]
        assigned = {}
        new_best_vset = []
        for _ in equi_sites:
            new_best_vset.append([])
        for ival, val in enumerate(self._best_vset):
            new_best_vset[attrib[ival]].append(val)
        for val, sites in zip(new_best_vset, equi_sites):
            for site in sites:
                assigned[site] = val
        return [[int(frac_site) for frac_site in assigned[site]] for site in structure]
    raise ValueError('Valences cannot be assigned!')