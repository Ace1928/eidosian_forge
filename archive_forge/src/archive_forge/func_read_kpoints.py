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
def read_kpoints(self):
    """ Reader of the .KP files """
    fname = self.getpath(ext='KP')
    try:
        with open(fname, 'r') as fd:
            nkp = int(next(fd))
            kpoints = np.empty((nkp, 3))
            kweights = np.empty(nkp)
            for i in range(nkp):
                line = next(fd)
                tokens = line.split()
                numbers = np.array(tokens[1:]).astype(float)
                kpoints[i] = numbers[:3]
                kweights[i] = numbers[3]
    except IOError:
        return 1
    self.results['kpoints'] = kpoints
    self.results['kweights'] = kweights
    return 0