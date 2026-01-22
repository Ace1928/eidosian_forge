from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
def get_lattice_constant(self):
    """Get the lattice constant of an element with cubic crystal structure."""
    if _refstate[self.atomicnumber]['symmetry'] != self.xtal_name:
        raise ValueError(('Cannot guess the %s lattice constant of' + ' an element with crystal structure %s.') % (self.xtal_name, _refstate[self.atomicnumber]['symmetry']))
    return _refstate[self.atomicnumber]['a']