from abc import ABC, abstractmethod
from typing import Mapping, Any
def get_potential_energies(self, atoms=None):
    return self.get_property('energies', atoms)