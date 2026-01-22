import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
def set_cell(self, atoms, change=False):
    lammps_cell, self.coord_transform = convert_cell(atoms.get_cell())
    xhi, xy, xz, _, yhi, yz, _, _, zhi = convert(lammps_cell.flatten(order='C'), 'distance', 'ASE', self.units)
    box_hi = [xhi, yhi, zhi]
    if change:
        cell_cmd = 'change_box all     x final 0 {} y final 0 {} z final 0 {}      xy final {} xz final {} yz final {} units box'.format(xhi, yhi, zhi, xy, xz, yz)
        if self.parameters.post_changebox_cmds is not None:
            for cmd in self.parameters.post_changebox_cmds:
                self.lmp.command(cmd)
    else:
        if self.parameters.create_box:
            self.lmp.command('box tilt large')
        lammps_boundary_conditions = self.lammpsbc(atoms).split()
        if 's' in lammps_boundary_conditions:
            pos = atoms.get_positions()
            if self.coord_transform is not None:
                pos = np.dot(self.coord_transform, pos.transpose())
                pos = pos.transpose()
            posmin = np.amin(pos, axis=0)
            posmax = np.amax(pos, axis=0)
            for i in range(0, 3):
                if lammps_boundary_conditions[i] == 's':
                    box_hi[i] = 1.05 * abs(posmax[i] - posmin[i])
        cell_cmd = 'region cell prism    0 {} 0 {} 0 {}     {} {} {}     units box'.format(*box_hi, xy, xz, yz)
    self.lmp.command(cell_cmd)