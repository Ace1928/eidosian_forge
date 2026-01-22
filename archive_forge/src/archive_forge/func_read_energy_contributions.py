import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def read_energy_contributions(self):
    """Reads the different energy contributions."""
    lines = self._outmol_lines()
    energies = dict()
    for n, line in enumerate(lines):
        if line.startswith('Energy components'):
            m = n + 1
            while not lines[m].strip() == '':
                energies[lines[m].split('=')[0].strip()] = float(re.findall('[-+]?\\d*\\.\\d+|\\d+', lines[m])[0]) * Hartree
                m += 1
    return energies