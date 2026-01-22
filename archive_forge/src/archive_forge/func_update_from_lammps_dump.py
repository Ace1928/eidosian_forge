import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def update_from_lammps_dump(self, fileobj, check=True):
    atoms = read(fileobj, format='lammps-dump')
    if len(atoms) != len(self):
        raise RuntimeError('Structure in ' + str(fileobj) + ' has wrong length: %d != %d' % (len(atoms), len(self)))
    if check:
        for a, b in zip(self, atoms):
            if not a.tag + 1 == b.number:
                raise RuntimeError('Atoms index %d are of different type (%d != %d)' % (a.index, a.tag + 1, b.number))
    self.set_cell(atoms.get_cell())
    self.set_positions(atoms.get_positions())
    if atoms.get_velocities() is not None:
        self.set_velocities(atoms.get_velocities())