from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pymatgen.analysis.elasticity.strain import Deformation, Strain
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
@property
def total_energy(self):
    """Total energy of this match."""
    return self.ground_state_energy + self.elastic_energy