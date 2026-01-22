from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
@due.dcite(Doi('10.1051/epjconf/20122200010'), description='Symmetry and magnetic structures')
def operate_magmom(self, magmom):
    """Apply time reversal operator on the magnetic moment. Note that
        magnetic moments transform as axial vectors, not polar vectors.

        See 'Symmetry and magnetic structures', Rodríguez-Carvajal and
        Bourée for a good discussion. DOI: 10.1051/epjconf/20122200010

        Args:
            magmom: Magnetic moment as electronic_structure.core.Magmom
            class or as list or np array-like

        Returns:
            Magnetic moment after operator applied as Magmom class
        """
    magmom = Magmom(magmom)
    transformed_moment = self.apply_rotation_only(magmom.global_moment) * np.linalg.det(self.rotation_matrix) * self.time_reversal
    return Magmom.from_global_moment_and_saxis(transformed_moment, magmom.saxis)