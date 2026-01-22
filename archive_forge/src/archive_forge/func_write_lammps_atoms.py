import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def write_lammps_atoms(self, atoms, connectivities):
    """Write atoms input for LAMMPS"""
    with open(self.prefix + '_atoms', 'w') as fileobj:
        self._write_lammps_atoms(fileobj, atoms, connectivities)