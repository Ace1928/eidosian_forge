import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
def initialise_lammps(self, atoms):
    if self.parameters.boundary:
        for cmd in self.parameters.lmpcmds:
            if 'boundary' in cmd:
                break
        else:
            self.lmp.command('boundary ' + self.lammpsbc(atoms))
    self.set_cell(atoms, change=not self.parameters.create_box)
    if self.parameters.atom_types is None:
        s = atoms.get_chemical_symbols()
        _, idx = np.unique(s, return_index=True)
        s_red = np.array(s)[np.sort(idx)].tolist()
        self.parameters.atom_types = {j: i + 1 for i, j in enumerate(s_red)}
    if self.parameters.create_box:
        n_types = len(self.parameters.atom_types)
        create_box_command = 'create_box {} cell'.format(n_types)
        self.lmp.command(create_box_command)
    if self.parameters.create_atoms:
        self.lmp.command('echo none')
        self.rebuild(atoms)
        self.lmp.command('echo log')
    else:
        self.previous_atoms_numbers = atoms.numbers.copy()
    for cmd in self.parameters.lmpcmds:
        self.lmp.command(cmd)
    for sym in self.parameters.atom_types:
        if self.parameters.atom_type_masses is None:
            mass = ase_atomic_masses[ase_atomic_numbers[sym]]
        else:
            mass = self.parameters.atom_type_masses[sym]
        self.lmp.command('mass %d %.30f' % (self.parameters.atom_types[sym], convert(mass, 'mass', 'ASE', self.units)))
    self.lmp.command('variable pxx equal pxx')
    self.lmp.command('variable pyy equal pyy')
    self.lmp.command('variable pzz equal pzz')
    self.lmp.command('variable pxy equal pxy')
    self.lmp.command('variable pxz equal pxz')
    self.lmp.command('variable pyz equal pyz')
    self.lmp.command('thermo_style custom pe pxx emol ecoul')
    self.lmp.command('variable fx atom fx')
    self.lmp.command('variable fy atom fy')
    self.lmp.command('variable fz atom fz')
    self.lmp.command('variable pe equal pe')
    self.lmp.command('neigh_modify delay 0 every 1 check yes')
    self.initialized = True