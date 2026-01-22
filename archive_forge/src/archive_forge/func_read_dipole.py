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
def read_dipole(self):
    """Read dipole moment. """
    dipole = np.zeros([1, 3])
    with open(self.getpath(ext='out'), 'r') as fd:
        for line in fd:
            if line.rfind('Electric dipole (Debye)') > -1:
                dipole = np.array([float(f) for f in line.split()[5:8]])
    self.results['dipole'] = dipole * 0.2081943482534