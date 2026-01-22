from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParams, get_neighbors_of_site_with_index
from pymatgen.core import Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def mu_so(species: str | Species, motif: Literal['oct', 'tet'], spin_state: Literal['high', 'low']) -> float | None:
    """Calculates the spin-only magnetic moment for a
        given species. Only supports transition metals.

        Args:
            species: Species
            motif ("oct" | "tet"): Tetrahedron or octahedron crystal site coordination
            spin_state ("low" | "high"): Whether the species is in a high or low spin state

        Returns:
            float: Spin-only magnetic moment in Bohr magnetons or None if
                species crystal field not defined
        """
    try:
        sp = get_el_sp(species)
        n = sp.get_crystal_field_spin(coordination=motif, spin_config=spin_state)
        return np.sqrt(n * (n + 2))
    except AttributeError:
        return None