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
def read_eigenvalues(self):
    """ A robust procedure using the suggestion by Federico Marchesin """
    fname = self.getpath(ext='EIG')
    try:
        with open(fname, 'r') as fd:
            self.results['fermi_energy'] = float(fd.readline())
            n, nspin, nkp = map(int, fd.readline().split())
            _ee = np.split(np.array(fd.read().split()).astype(float), nkp)
    except IOError:
        return 1
    ksn2e = np.delete(_ee, 0, 1).reshape([nkp, nspin, n])
    eigarray = np.empty((nspin, nkp, n))
    eigarray[:] = np.inf
    for k, sn2e in enumerate(ksn2e):
        for s, n2e in enumerate(sn2e):
            eigarray[s, k, :] = n2e
    assert np.isfinite(eigarray).all()
    self.results['eigenvalues'] = eigarray
    return 0