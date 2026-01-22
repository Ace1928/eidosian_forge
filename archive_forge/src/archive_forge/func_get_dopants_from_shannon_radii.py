from __future__ import annotations
import warnings
import numpy as np
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.core import Element, Species
def get_dopants_from_shannon_radii(bonded_structure, num_dopants=5, match_oxi_sign=False):
    """
    Get dopant suggestions based on Shannon radii differences.

    Args:
        bonded_structure (StructureGraph): A pymatgen structure graph
            decorated with oxidation states. For example, generated using the
            CrystalNN.get_bonded_structure() method.
        num_dopants (int): The number of suggestions to return for
            n- and p-type dopants.
        match_oxi_sign (bool): Whether to force the dopant and original species
            to have the same sign of oxidation state. E.g. If the original site
            is in a negative charge state, then only negative dopants will be
            returned.

    Returns:
        dict: Dopant suggestions, given as a dictionary with keys "n_type" and
            "p_type". The suggestions for each doping type are given as a list of
            dictionaries, each with they keys:

            - "radii_diff": The difference between the Shannon radii of the species.
            - "dopant_species": The dopant species.
            - "original_species": The substituted species.
    """
    all_species = [Species(el, oxi) for el in Element for oxi in el.common_oxidation_states]
    cn_and_species = {(bonded_structure.get_coordination_of_site(idx), bonded_structure.structure[idx].specie) for idx in range(len(bonded_structure))}
    cn_to_radii_map = {}
    possible_dopants = []
    for cn, species in cn_and_species:
        cn_roman = _int_to_roman(cn)
        try:
            species_radius = species.get_shannon_radius(cn_roman)
        except KeyError:
            warnings.warn(f'Shannon radius not found for {species} with coordination number {cn}.\nSkipping...')
            continue
        if cn not in cn_to_radii_map:
            cn_to_radii_map[cn] = _shannon_radii_from_cn(all_species, cn_roman, radius_to_compare=species_radius)
        shannon_radii = cn_to_radii_map[cn]
        possible_dopants += [{'radii_diff': p['radii_diff'], 'dopant_species': p['species'], 'original_species': species} for p in shannon_radii]
    possible_dopants.sort(key=lambda x: abs(x['radii_diff']))
    return _get_dopants(possible_dopants, num_dopants, match_oxi_sign)