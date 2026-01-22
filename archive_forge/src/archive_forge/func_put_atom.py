import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def put_atom(self, point):
    """Place an atom given its integer coordinates."""
    if self.bravais_basis is None:
        pos = np.dot(point, self.crystal_basis)
        if self.debug >= 2:
            print('Placing an atom at (%d,%d,%d) ~ (%.3f, %.3f, %.3f).' % (tuple(point) + tuple(pos)))
        self.atoms[self.nput] = pos
        self.elements[self.nput] = self.atomicnumber
        self.nput += 1
    else:
        for i, offset in enumerate(self.natural_bravais_basis):
            pos = np.dot(point + offset, self.crystal_basis)
            if self.debug >= 2:
                print('Placing an atom at (%d+%f, %d+%f, %d+%f) ~ (%.3f, %.3f, %.3f).' % (point[0], offset[0], point[1], offset[1], point[2], offset[2], pos[0], pos[1], pos[2]))
            self.atoms[self.nput] = pos
            if self.element_basis is None:
                self.elements[self.nput] = self.atomicnumber
            else:
                self.elements[self.nput] = self.atomicnumber[i]
            self.nput += 1