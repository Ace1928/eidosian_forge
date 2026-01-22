import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def read_potential(self, filename):
    """Reads a LAMMPS EAM file in alloy or adp format
        and creates the interpolation functions from the data
        """
    if isinstance(filename, str):
        with open(filename) as fd:
            self._read_potential(fd)
    else:
        fd = filename
        self._read_potential(fd)