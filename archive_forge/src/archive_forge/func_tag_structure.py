from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParams, get_neighbors_of_site_with_index
from pymatgen.core import Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def tag_structure(self, structure: Structure, calculate_valences: bool=True, guesstimate_spin: bool=False, op_threshold: float=0.1) -> Structure:
    """
        Convenience method, uses get_analysis_and_structure method.
        Add a "possible_jt_active" site property on Structure.

        Args:
            structure: input structure
            calculate_valences: whether to attempt to calculate valences or not, structure
                should have oxidation states to perform analysis (Default value = True)
            guesstimate_spin: whether to guesstimate spin state from magnetic moments
                or not, use with caution (Default value = False)
            op_threshold: threshold for order parameter above which to consider site
                to match an octahedral or tetrahedral motif, since Jahn-Teller structures
                can often be
                quite distorted, this threshold is smaller than one might expect

        Returns:
            Decorated Structure, will be in primitive setting.
        """
    try:
        analysis, structure = self.get_analysis_and_structure(structure, calculate_valences=calculate_valences, guesstimate_spin=guesstimate_spin, op_threshold=op_threshold)
        jt_sites = [False] * len(structure)
        if analysis['active']:
            for site in analysis['sites']:
                for index in site['site_indices']:
                    jt_sites[index] = True
                    structure.add_site_property('possible_jt_active', jt_sites)
        return structure
    except Exception as exc:
        warnings.warn(f'Error analyzing {structure.reduced_formula}: {exc}')
        return structure