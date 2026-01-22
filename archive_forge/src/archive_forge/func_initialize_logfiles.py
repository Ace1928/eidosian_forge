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
def initialize_logfiles(self, logfile=None, eigenmode_logfile=None):
    self.logfile = self.openfile(logfile, comm=world)
    self.eigenmode_logfile = self.openfile(eigenmode_logfile, comm=world)