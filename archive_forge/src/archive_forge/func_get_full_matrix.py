from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
@staticmethod
def get_full_matrix(thermal_displacement: ArrayLike[ArrayLike]) -> np.ndarray[np.ndarray]:
    """Transfers the reduced matrix to the full matrix (order of reduced matrix U11, U22, U33, U23, U13, U12).

        Args:
            thermal_displacement: 2d numpy array, first dimension are the atoms

        Returns:
            3d numpy array including thermal displacements, first dimensions are the atoms
        """
    matrixform = np.zeros((len(thermal_displacement), 3, 3))
    for imat, mat in enumerate(thermal_displacement):
        matrixform[imat][0][0] = mat[0]
        matrixform[imat][1][1] = mat[1]
        matrixform[imat][2][2] = mat[2]
        matrixform[imat][1][2] = mat[3]
        matrixform[imat][2][1] = mat[3]
        matrixform[imat][0][2] = mat[4]
        matrixform[imat][2][0] = mat[4]
        matrixform[imat][0][1] = mat[5]
        matrixform[imat][1][0] = mat[5]
    return matrixform