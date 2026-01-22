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
def refine_rotation(self):
    """Helper method for refining rotation matrix by ensuring
        that second and third rows are perpendicular to the first.
        Gets new y vector from an orthogonal projection of x onto y
        and the new z vector from a cross product of the new x and y.

        Args:
            tol to test for rotation

        Returns:
            new rotation matrix
        """
    new_x, y = (get_uvec(self[0]), get_uvec(self[1]))
    new_y = y - np.dot(new_x, y) * new_x
    new_z = np.cross(new_x, new_y)
    return SquareTensor([new_x, new_y, new_z])