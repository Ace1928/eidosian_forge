import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError, Parameters
from ase.units import kcal, mol, Debye
def get_homo_lumo_levels(self):
    eigs = self.eigenvalues
    if self.nspins == 1:
        nocc = self.no_occ_levels
        return np.array([eigs[0, 0, nocc - 1], eigs[0, 0, nocc]])
    else:
        na = self.no_alpha_electrons
        nb = self.no_beta_electrons
        if na == 0:
            return (None, self.eigenvalues[1, 0, nb - 1])
        elif nb == 0:
            return (self.eigenvalues[0, 0, na - 1], None)
        else:
            eah, eal = eigs[0, 0, na - 1:na + 1]
            ebh, ebl = eigs[1, 0, nb - 1:nb + 1]
            return np.array([max(eah, ebh), min(eal, ebl)])