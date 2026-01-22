import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from scipy.special import erfinv, erfc
from ase.neighborlist import neighbor_list
from ase.parallel import world
from ase.utils import IOContext
@property
def implemented_properties(self):
    return self.calculator.implemented_properties