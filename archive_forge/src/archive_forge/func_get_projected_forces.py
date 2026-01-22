import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def get_projected_forces(self, pos=None):
    """Return the projected forces."""
    if pos is not None:
        forces = self.get_forces(real=True, pos=pos).copy()
    else:
        forces = self.forces0.copy()
    for k, mode in enumerate(self.eigenmodes):
        if self.get_curvature(order=k) > 0.0 and self.order == 1:
            forces = -parallel_vector(forces, mode)
        else:
            forces -= 2 * parallel_vector(forces, mode)
    return forces