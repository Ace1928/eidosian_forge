from math import pi
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.units import Bohr, Ha
def make_test_dft_calculation():
    a = b = 2.0
    c = 6.0
    atoms = Atoms(positions=[(0, 0, c / 2)], symbols='H', pbc=(1, 1, 0), cell=(a, b, c), calculator=TestCalculator())
    return atoms