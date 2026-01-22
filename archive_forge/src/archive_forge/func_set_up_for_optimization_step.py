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
def set_up_for_optimization_step(self):
    """At the end of rotation, prepare for displacement of the dimer."""
    self.atoms.set_positions(self.pos0)
    self.forces1E = None