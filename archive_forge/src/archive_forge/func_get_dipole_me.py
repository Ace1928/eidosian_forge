import numpy as np
from ase.units import Hartree, Bohr
def get_dipole_me(self, form='r'):
    """Return the excitations dipole matrix element
        including the occupation factor sqrt(fij)"""
    if form == 'r':
        me = -self.mur
    elif form == 'v':
        me = -self.muv
    else:
        raise RuntimeError('Unknown form >' + form + '<')
    return np.sqrt(self.fij) * me