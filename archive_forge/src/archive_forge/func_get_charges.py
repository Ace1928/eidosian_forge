from abc import ABC, abstractmethod
from typing import Mapping, Any
def get_charges(self, atoms=None):
    return self.get_property('charges', atoms)