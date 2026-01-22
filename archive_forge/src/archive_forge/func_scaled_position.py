import numpy as np
from ase.data import atomic_numbers, chemical_symbols, atomic_masses
@scaled_position.setter
def scaled_position(self, value):
    pos = self.atoms.cell.cartesian_positions(value)
    self.position = pos