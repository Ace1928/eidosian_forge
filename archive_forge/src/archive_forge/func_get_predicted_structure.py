from __future__ import annotations
import os
import warnings
import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
def get_predicted_structure(self, structure: Structure, icsd_vol=False):
    """
        Given a structure, returns back the structure scaled to predicted
        volume.

        Args:
            structure (Structure): structure w/unknown volume

        Returns:
            a Structure object with predicted volume
        """
    new_structure = structure.copy()
    new_structure.scale_lattice(self.predict(structure, icsd_vol=icsd_vol))
    return new_structure