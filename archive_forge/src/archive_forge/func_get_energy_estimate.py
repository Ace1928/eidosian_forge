import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def get_energy_estimate(self, D, Gbar, b):
    de = 0.0
    for n in range(len(D)):
        de += D[n] * Gbar[n] + 0.5 * D[n] * b[n] * D[n]
    return de