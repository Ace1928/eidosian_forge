import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def get_strain_rate(self):
    """Get the strain rate as an upper-triangular 3x3 matrix.

        This includes the fluctuations in the shape of the computational box.

        """
    return np.array(self.eta, copy=1)