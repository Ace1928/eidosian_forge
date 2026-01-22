from ase.io import Trajectory
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.calculators.singlepoint import SinglePointCalculator
import ase.parallel as mpi
import numpy as np
import shutil
import os
import types
from math import log
from math import exp
from contextlib import ExitStack
def get_energies_one_image(self, image):
    """Utility method to extract energy of an image and return np.NaN
        if invalid."""
    try:
        energy = image.get_potential_energy()
    except RuntimeError:
        energy = np.NaN
    return energy