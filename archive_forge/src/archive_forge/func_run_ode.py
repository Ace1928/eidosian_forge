import sys
import threading
import warnings
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
import ase.parallel
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.geometry import find_mic
from ase.utils import lazyproperty, deprecated
from ase.utils.forcecurve import fit_images
from ase.optimize.precon import Precon, PreconImages
from ase.optimize.ode import ode12r
def run_ode(self, fmax):
    try:
        ode12r(self.force_function, self.neb.get_positions().reshape(-1), fmax=fmax, rtol=self.rtol, C1=self.C1, C2=self.C2, steps=self.max_steps, verbose=self.verbose, callback=self.callback, residual=self.get_residual)
        return True
    except OptimizerConvergenceError:
        return False