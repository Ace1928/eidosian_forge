from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParams, get_neighbors_of_site_with_index
from pymatgen.core import Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_magnitude_of_effect_from_species(self, species: str | Species, spin_state: str, motif: str) -> str:
    """
        Get magnitude of Jahn-Teller effect from provided species, spin state and motif.

        Args:
            species: e.g. Fe2+
            spin_state: "high" or "low"
            motif: "oct" or "tet"

        Returns:
            str: "none", "weak" or "strong"
        """
    magnitude = 'none'
    sp = get_el_sp(species)
    if isinstance(sp, Species) and sp.element.is_transition_metal:
        d_electrons = self._get_number_of_d_electrons(sp)
        if motif in self.spin_configs:
            if spin_state not in self.spin_configs[motif][d_electrons]:
                spin_state = self.spin_configs[motif][d_electrons]['default']
            spin_config = self.spin_configs[motif][d_electrons][spin_state]
            magnitude = JahnTellerAnalyzer.get_magnitude_of_effect_from_spin_config(motif, spin_config)
    else:
        warnings.warn('No data for this species.')
    return magnitude