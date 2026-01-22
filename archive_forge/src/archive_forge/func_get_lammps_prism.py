import numpy as np
from ase.geometry import wrap_positions
def get_lammps_prism(self):
    """Return into lammps coordination system rotated cell

        :returns: lammps cell
        :rtype: np.array

        """
    return self.lammps_cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]