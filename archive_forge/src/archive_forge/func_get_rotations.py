import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def get_rotations(self):
    """Return all rotations, including inversions for
        centrosymmetric crystals."""
    if self.centrosymmetric:
        return np.vstack((self.rotations, -self.rotations))
    else:
        return self.rotations