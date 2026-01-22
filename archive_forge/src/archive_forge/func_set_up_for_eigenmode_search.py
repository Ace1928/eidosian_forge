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
def set_up_for_eigenmode_search(self):
    """Before eigenmode search, prepare for rotation."""
    self.pos0 = self.atoms.get_positions()
    self.update_center_forces()
    self.update_virtual_positions()
    self.control.reset_counter('rotcount')
    self.forces1E = None