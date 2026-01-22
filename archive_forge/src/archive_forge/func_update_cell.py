import numpy as np
from ase.geometry import wrap_positions
def update_cell(self, lammps_cell):
    """Rotate new lammps cell into ase coordinate system

        :param lammps_cell: new lammps cell received after executing lammps
        :returns: ase cell
        :rtype: np.array

        """
    self.lammps_cell = lammps_cell
    self.lammps_tilt = self.lammps_cell.copy()
    for iteri, (i, j, k) in enumerate(FLIP_ORDER):
        if self.flip[iteri]:
            change = self.lammps_cell[k][k]
            change *= np.sign(self.lammps_cell[i][j])
            self.lammps_tilt[i][j] -= change
    new_ase_cell = np.dot(self.lammps_tilt, self.rot_mat.T)
    new_vol = np.linalg.det(new_ase_cell)
    old_vol = np.linalg.det(self.ase_cell)
    test_residual = self.ase_cell.copy()
    test_residual *= (new_vol / old_vol) ** (1.0 / 3.0)
    test_residual -= new_ase_cell
    if any(np.linalg.norm(test_residual, axis=1) > 0.5 * np.linalg.norm(self.ase_cell, axis=1)):
        print('WARNING: Significant simulation cell changes from LAMMPS ' + 'detected.\n' + ' ' * 9 + 'Backtransformation to ASE might fail!')
    return new_ase_cell