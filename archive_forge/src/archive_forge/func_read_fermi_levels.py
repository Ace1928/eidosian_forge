import os
import numpy as np
from ase.calculators.calculator import (FileIOCalculator, kpts2ndarray,
from ase.units import Hartree, Bohr
def read_fermi_levels(self):
    """ Read Fermi level(s) from dftb output file (results.tag). """
    for iline, line in enumerate(self.lines):
        fstring = 'fermi_level   '
        if line.find(fstring) >= 0:
            index_fermi = iline + 1
            break
    else:
        return None
    fermi_levels = []
    words = self.lines[index_fermi].split()
    assert len(words) in [1, 2], 'Expected either 1 or 2 Fermi levels'
    for word in words:
        e = float(word)
        if abs(e) > 1e-08:
            fermi_levels.append(e)
    return np.array(fermi_levels) * Hartree