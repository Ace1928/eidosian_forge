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
def update_eigenmode(self, eigenmode):
    """Update the eigenmode in the MinModeAtoms object."""
    self.eigenmode = eigenmode
    self.update_virtual_positions()
    self.control.increment_counter('rotcount')