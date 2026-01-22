import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_scf_kgrid(atoms, parameters):
    kpts, scf_kgrid = (parameters.get('kpts'), parameters.get('scf_kgrid'))
    if isinstance(kpts, (tuple, list, np.ndarray)) and len(kpts) == 3 and isinstance(kpts[0], int):
        return kpts
    elif isinstance(kpts, float) or isinstance(kpts, int):
        return tuple(kpts2sizeandoffsets(atoms=atoms, density=kpts)[0])
    else:
        return scf_kgrid