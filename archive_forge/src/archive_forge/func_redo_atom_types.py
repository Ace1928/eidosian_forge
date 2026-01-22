import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
def redo_atom_types(self, atoms):
    current_types = set(((i + 1, self.parameters.atom_types[sym]) for i, sym in enumerate(atoms.get_chemical_symbols())))
    try:
        previous_types = set(((i + 1, self.parameters.atom_types[ase_chemical_symbols[Z]]) for i, Z in enumerate(self.previous_atoms_numbers)))
    except Exception:
        previous_types = set()
    for i, i_type in current_types - previous_types:
        cmd = 'set atom {} type {}'.format(i, i_type)
        self.lmp.command(cmd)
    self.previous_atoms_numbers = atoms.numbers.copy()