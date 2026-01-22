from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def structure_transform(self, original_structure, new_structure, refine_rotation=True):
    """Transforms a tensor from one basis for an original structure
        into a new basis defined by a new structure.

        Args:
            original_structure (Structure): structure corresponding
                to the basis of the current tensor
            new_structure (Structure): structure corresponding to the
                desired basis
            refine_rotation (bool): whether to refine the rotations
                generated in get_ieee_rotation

        Returns:
            Tensor that has been transformed such that its basis
            corresponds to the new_structure's basis
        """
    sm = StructureMatcher()
    if not sm.fit(original_structure, new_structure):
        warnings.warn('original and new structures do not match!')
    trans_1 = self.get_ieee_rotation(original_structure, refine_rotation)
    trans_2 = self.get_ieee_rotation(new_structure, refine_rotation)
    new = self.rotate(trans_1)
    return new.rotate(np.transpose(trans_2))