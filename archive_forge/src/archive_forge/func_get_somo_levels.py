import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError, Parameters
from ase.units import kcal, mol, Debye
def get_somo_levels(self):
    assert self.nspins == 2
    na, nb = (self.no_alpha_electrons, self.no_beta_electrons)
    if na == 0:
        return (None, self.eigenvalues[1, 0, nb - 1])
    elif nb == 0:
        return (self.eigenvalues[0, 0, na - 1], None)
    else:
        return np.array([self.eigenvalues[0, 0, na - 1], self.eigenvalues[1, 0, nb - 1]])