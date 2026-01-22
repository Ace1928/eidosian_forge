import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_atoms_unitvectors(atoms, parameters):
    zero_vec = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    if np.all(atoms.get_cell() == zero_vec) is True:
        default_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return parameters.get('atoms_unitvectors', default_cell)
    unit = parameters.get('atoms_unitvectors_unit', 'ang').lower()
    if unit == 'ang':
        atoms_unitvectors = atoms.get_cell()
    elif unit == 'au':
        atoms_unitvectors = atoms.get_cell() / Bohr
    return atoms_unitvectors