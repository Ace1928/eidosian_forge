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
@staticmethod
def inversion(origin: ArrayLike=(0, 0, 0)) -> SymmOp:
    """Inversion symmetry operation about axis.

        Args:
            origin (3x1 array): Origin of the inversion operation. Defaults
                to [0, 0, 0].

        Returns:
            SymmOp representing an inversion operation about the origin.
        """
    mat = -np.eye(4)
    mat[3, 3] = 1
    mat[0:3, 3] = 2 * np.array(origin)
    return SymmOp(mat)