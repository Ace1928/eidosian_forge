from __future__ import annotations
import functools
import itertools
import logging
from operator import mul
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.alchemy.filters import RemoveDuplicatesFilter, RemoveExistingFilter
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
from pymatgen.core import get_el_sp
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.util.due import Doi, due
def pred_from_list(self, species_list):
    """
        There are an exceptionally large number of substitutions to
        look at (260^n), where n is the number of species in the
        list. We need a more efficient than brute force way of going
        through these possibilities. The brute force method would be:

            output = []
            for p in itertools.product(self._sp.species_list, repeat=len(species_list)):
                if self._sp.conditional_probability_list(p, species_list) > self._threshold:
                    output.append(dict(zip(species_list, p)))
            return output

        Instead of that we do a branch and bound.

        Args:
            species_list:
                list of species in the starting structure

        Returns:
            list of dictionaries, each including a substitutions
            dictionary, and a probability value
        """
    species_list = [get_el_sp(sp) for sp in species_list]
    max_probabilities = []
    for s2 in species_list:
        max_p = 0
        for s1 in self._sp.species:
            max_p = max([self._sp.cond_prob(s1, s2), max_p])
        max_probabilities.append(max_p)
    output = []

    def _recurse(output_prob, output_species):
        best_case_prob = list(max_probabilities)
        best_case_prob[:len(output_prob)] = output_prob
        if functools.reduce(mul, best_case_prob) > self._threshold:
            if len(output_species) == len(species_list):
                odict = {'substitutions': dict(zip(species_list, output_species)), 'probability': functools.reduce(mul, best_case_prob)}
                output.append(odict)
                return
            for sp in self._sp.species:
                i = len(output_prob)
                prob = self._sp.cond_prob(sp, species_list[i])
                _recurse([*output_prob, prob], [*output_species, sp])
    _recurse([], [])
    logging.info(f'{len(output)} substitutions found')
    return output