from math import cos, sin, pi
import numpy as np
from ase import Atoms
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0, rOH, angleHOH
from ase.calculators.qmmm import SimpleQMMM, EIQMMM, LJInteractions
from ase.calculators.turbomole import Turbomole
from ase.constraints import FixInternals
from ase.optimize import BFGS
Test the Turbomole calculator in simple QMMM and
    explicit interaction QMMM simulations.