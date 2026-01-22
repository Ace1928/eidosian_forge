import numpy as np
from ase.atoms import Atoms
def set_quaternions(self, quaternions):
    self.set_array('quaternions', quaternions, quaternion=(4,))