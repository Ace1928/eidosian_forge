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
def pred_from_structures(self, target_species, structures_list, remove_duplicates=True, remove_existing=False):
    """
        Performs a structure prediction targeting compounds containing all of
        the target_species, based on a list of structure (those structures
        can for instance come from a database like the ICSD). It will return
        all the structures formed by ionic substitutions with a probability
        higher than the threshold.

        Notes:
            If the default probability model is used, input structures must
            be oxidation state decorated. See AutoOxiStateDecorationTransformation

            This method does not change the number of species in a structure. i.e
            if the number of target species is 3, only input structures containing
            3 species will be considered.

        Args:
            target_species:
                a list of species with oxidation states
                e.g., [Species('Li+'), Species('Ni2+'), Species('O-2')]

            structures_list:
                list of dictionary of the form {'structure': Structure object, 'id': some id where it comes from}
                The id can for instance refer to an ICSD id.

            remove_duplicates:
                if True, the duplicates in the predicted structures will
                be removed

            remove_existing:
                if True, the predicted structures that already exist in the
                structures_list will be removed

        Returns:
            a list of TransformedStructure objects.
        """
    target_species = [get_el_sp(sp) for sp in target_species]
    result = []
    transmuter = StandardTransmuter([])
    if len(set(target_species) & set(self.get_allowed_species())) != len(target_species):
        raise ValueError('the species in target_species are not allowed for the probability model you are using')
    for permutation in itertools.permutations(target_species):
        for s in structures_list:
            els = s['structure'].elements
            if len(els) == len(permutation) and len(set(els) & set(self.get_allowed_species())) == len(els) and (self._sp.cond_prob_list(permutation, els) > self._threshold):
                clean_subst = {els[i]: permutation[i] for i in range(len(els)) if els[i] != permutation[i]}
                if len(clean_subst) == 0:
                    continue
                transf = SubstitutionTransformation(clean_subst)
                if Substitutor._is_charge_balanced(transf.apply_transformation(s['structure'])):
                    ts = TransformedStructure(s['structure'], [transf], history=[{'source': s['id']}], other_parameters={'type': 'structure_prediction', 'proba': self._sp.cond_prob_list(permutation, els)})
                    result.append(ts)
                    transmuter.append_transformed_structures([ts])
    if remove_duplicates:
        transmuter.apply_filter(RemoveDuplicatesFilter(symprec=self._symprec))
    if remove_existing:
        chemsys = {sp.symbol for sp in target_species}
        structures_list_target = [st['structure'] for st in structures_list if Substitutor._is_from_chemical_system(chemsys, st['structure'])]
        transmuter.apply_filter(RemoveExistingFilter(structures_list_target, symprec=self._symprec))
    return transmuter.transformed_structures