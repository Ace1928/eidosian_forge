import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.ga.utilities import (atoms_too_close, atoms_too_close_two_sets,
from ase.ga.offspring_creator import OffspringCreator
def to_use(self):
    """Tells whether this position is at the right side."""
    if self.distance > 0.0 and self.origin == 0:
        return True
    elif self.distance < 0.0 and self.origin == 1:
        return True
    else:
        return False