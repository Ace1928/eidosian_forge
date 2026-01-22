from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
def check_basis_volume(self):
    """Check the volume of the unit cell."""
    vol1 = abs(np.linalg.det(self.basis))
    cellsize = self.atoms_in_unit_cell
    if self.bravais_basis is not None:
        cellsize *= len(self.bravais_basis)
    vol2 = self.calc_num_atoms() * self.latticeconstant ** 3 / cellsize
    assert abs(vol1 - vol2) < 1e-05