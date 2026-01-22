import numpy as np
from ase.geometry import wrap_positions
def vector_to_ase(self, vec, wrap=False):
    """Rotate vector from lammps coordinate system to ase one

        :param vec: to be rotated lammps-vector
        :param wrap: was vector wrapped into 'lammps_cell'
        :returns: ase-vector
        :rtype: np.array

        """
    if wrap:
        translate = np.linalg.solve(self.lammps_tilt.T, self.lammps_cell.T).T
        fractional = np.linalg.solve(self.lammps_tilt.T, vec.T).T
        for ifrac in fractional:
            for zyx in reversed(range(3)):
                if ifrac[zyx] >= 1.0 and self.pbc[zyx]:
                    ifrac -= translate[zyx]
                elif ifrac[zyx] < 0.0 and self.pbc[zyx]:
                    ifrac += translate[zyx]
        vec = np.dot(fractional, self.lammps_tilt)
    return np.dot(vec, self.rot_mat.T)