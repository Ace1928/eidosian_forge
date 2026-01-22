import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def attach_atoms(self, atoms):
    """Assign atoms to a restored dynamics object.

        This function must be called to set the atoms immediately after the
        dynamics object has been read from a trajectory.
        """
    try:
        self.atoms
    except AttributeError:
        pass
    else:
        raise RuntimeError('Cannot call attach_atoms on a dynamics which already has atoms.')
    MolecularDynamics.__init__(self, atoms, self.dt)
    limit = 1e-06
    h = self._getbox()
    if max(abs((h - self.h).ravel())) > limit:
        raise RuntimeError('The unit cell of the atoms does not match the unit cell stored in the file.')
    self.inv_h = linalg.inv(self.h)
    self.q = np.dot(self.atoms.get_positions(), self.inv_h) - 0.5
    self._calculate_q_past_and_future()
    self.initialized = 1