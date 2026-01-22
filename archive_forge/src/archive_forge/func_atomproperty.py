import numpy as np
from ase.data import atomic_numbers, chemical_symbols, atomic_masses
def atomproperty(name, doc):
    """Helper function to easily create Atom attribute property."""

    def getter(self):
        return self.get(name)

    def setter(self, value):
        self.set(name, value)

    def deleter(self):
        self.delete(name)
    return property(getter, setter, deleter, doc)