import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def lines_to_list(lines):
    """Make the data one long line so as not to care how its formatted
            """
    data = []
    for line in lines:
        data.extend(line.split())
    return data