import numpy as np
import ase.units as units
from ase.calculators.calculator import Calculator, all_changes
def redistribute_forces(self, forces):
    return forces