from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
def make_crystal_basis(self):
    """Make the basis matrix for the crystal unit cell and the system unit cell."""
    self.crystal_basis = self.latticeconstant * self.basis_factor * self.int_basis
    self.miller_basis = self.latticeconstant * np.identity(3)
    self.basis = np.dot(self.directions, self.crystal_basis)
    self.check_basis_volume()