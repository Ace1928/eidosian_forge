import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def get_hessian(self):
    if not hasattr(self, 'hessian'):
        self.set_default_hessian()
    return self.hessian