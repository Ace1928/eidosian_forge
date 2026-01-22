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
def find_eigenmodes(self, order=1):
    """Launch eigenmode searches."""
    if self.control.get_parameter('eigenmode_method').lower() != 'dimer':
        e = 'Only the Dimer control object has been implemented.'
        raise NotImplementedError(e)
    for k in range(order):
        if k > 0:
            self.ensure_eigenmode_orthogonality(k + 1)
        search = DimerEigenmodeSearch(self, self.control, eigenmode=self.eigenmodes[k], basis=self.eigenmodes[:k])
        search.converge_to_eigenmode()
        search.set_up_for_optimization_step()
        self.eigenmodes[k] = search.get_eigenmode()
        self.curvatures[k] = search.get_curvature()