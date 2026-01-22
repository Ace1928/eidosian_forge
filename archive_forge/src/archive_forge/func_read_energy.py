import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def read_energy(self):
    """Read energy from SIESTA's text-output file.
        """
    fname = self.getpath(ext='out')
    with open(fname, 'r') as fd:
        text = fd.read().lower()
    assert 'final energy' in text
    lines = iter(text.split('\n'))
    for line in lines:
        has_energy = line.startswith('siesta: etot    =')
        if has_energy:
            self.results['energy'] = float(line.split()[-1])
            line = next(lines)
            self.results['free_energy'] = float(line.split()[-1])
    if 'energy' not in self.results or 'free_energy' not in self.results:
        raise RuntimeError