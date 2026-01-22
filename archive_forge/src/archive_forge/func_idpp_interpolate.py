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
@deprecated("Please use NEB's interpolate(method='idpp') method or directly call the idpp_interpolate function from ase.neb")
def idpp_interpolate(self, traj='idpp.traj', log='idpp.log', fmax=0.1, optimizer=MDMin, mic=False, steps=100):
    idpp_interpolate(self, traj=traj, log=log, fmax=fmax, optimizer=optimizer, mic=mic, steps=steps)