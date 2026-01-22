import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def ekin(p, m=self.atoms.get_masses()):
    p2 = np.sum(p * p, -1)
    return 0.5 * np.sum(p2 / m) / len(m)