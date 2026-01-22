import numpy as np
from ase.data import atomic_numbers, chemical_symbols, atomic_masses
def xyzproperty(index):
    """Helper function to easily create Atom XYZ-property."""

    def getter(self):
        return self.position[index]

    def setter(self, value):
        self.position[index] = value
    return property(getter, setter, doc='XYZ'[index] + '-coordinate')