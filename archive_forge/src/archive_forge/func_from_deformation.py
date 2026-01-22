from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
@classmethod
def from_deformation(cls, deformation: ArrayLike) -> Self:
    """
        Factory method that returns a Strain object from a deformation
        gradient.

        Args:
            deformation (ArrayLike): 3x3 array defining the deformation
        """
    dfm = Deformation(deformation)
    return cls(0.5 * (np.dot(dfm.trans, dfm) - np.eye(3)))