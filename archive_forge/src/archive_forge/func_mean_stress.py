from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import SquareTensor
@property
def mean_stress(self):
    """Returns the mean stress."""
    return 1 / 3 * self.trace()