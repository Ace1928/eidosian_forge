import numpy as np
import ase.units as units
from ase.calculators.calculator import Calculator, all_changes
def get_virtual_charges(self, atoms):
    charges = np.empty(len(atoms))
    charges[:] = qH
    if atoms.numbers[0] == 8:
        charges[::3] = -2 * qH
    else:
        charges[2::3] = -2 * qH
    return charges