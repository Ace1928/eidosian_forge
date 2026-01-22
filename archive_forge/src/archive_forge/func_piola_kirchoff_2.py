from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import SquareTensor
def piola_kirchoff_2(self, def_grad):
    """
        Calculates the second Piola-Kirchoff stress.

        Args:
            def_grad (3x3 array-like): rate of deformation tensor
        """
    def_grad = SquareTensor(def_grad)
    if not self.is_symmetric:
        raise ValueError('The stress tensor is not symmetric, PK stress is based on a symmetric stress tensor.')
    return def_grad.det * np.dot(np.dot(def_grad.inv, self), def_grad.inv.trans)