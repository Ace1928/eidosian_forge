import os
import sys
import numpy as np
from ase import units
def get_ZPE_correction(self):
    """Returns the zero-point vibrational energy correction in eV."""
    zpe = 0.0
    for energy in self.vib_energies:
        zpe += 0.5 * energy
    return zpe