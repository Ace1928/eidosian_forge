from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import SquareTensor
@property
def von_mises(self):
    """Returns the von Mises stress."""
    if not self.is_symmetric():
        raise ValueError('The stress tensor is not symmetric, Von Mises stress is based on a symmetric stress tensor.')
    return math.sqrt(3 * self.dev_principal_invariants[1])