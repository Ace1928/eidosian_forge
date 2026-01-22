import numpy as np
from ase.data import atomic_numbers, chemical_symbols, atomic_masses
def cut_reference_to_atoms(self):
    """Cut reference to atoms object."""
    for name in names:
        self.data[name] = self.get_raw(name)
    self.index = None
    self.atoms = None