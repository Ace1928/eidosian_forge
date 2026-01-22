import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_spinpol(atoms, parameters):
    """ Judgeds the keyword 'scf.SpinPolarization'
     If the keyword is not None, spinpol gets the keyword by following priority
       1. standard_spinpol
       2. scf_spinpolarization
       3. magnetic moments of atoms
    """
    standard_spinpol = parameters.get('spinpol', None)
    scf_spinpolarization = parameters.get('scf_spinpolarization', None)
    m = atoms.get_initial_magnetic_moments()
    syn = {True: 'On', False: None, 'on': 'On', 'off': None, None: None, 'nc': 'NC'}
    spinpol = np.any(m >= 0.1)
    if scf_spinpolarization is not None:
        spinpol = scf_spinpolarization
    if standard_spinpol is not None:
        spinpol = standard_spinpol
    if isinstance(spinpol, str):
        spinpol = spinpol.lower()
    return syn[spinpol]