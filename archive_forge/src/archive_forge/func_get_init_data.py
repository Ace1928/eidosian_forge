import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def get_init_data(self):
    """Return the data needed to initialize a new NPT dynamics."""
    return {'dt': self.dt, 'temperature': self.temperature, 'desiredEkin': self.desiredEkin, 'externalstress': self.externalstress, 'mask': self.mask, 'ttime': self.ttime, 'tfact': self.tfact, 'pfactor_given': self.pfactor_given, 'pfact': self.pfact, 'frac_traceless': self.frac_traceless}