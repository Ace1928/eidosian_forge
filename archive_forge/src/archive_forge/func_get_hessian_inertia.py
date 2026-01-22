import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def get_hessian_inertia(self, eigenvalues):
    self.write_log('eigenvalues %2.2f %2.2f %2.2f ' % (eigenvalues[0], eigenvalues[1], eigenvalues[2]))
    n = 0
    while eigenvalues[n] < 0:
        n += 1
    return n