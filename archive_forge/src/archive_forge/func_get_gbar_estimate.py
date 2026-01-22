import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def get_gbar_estimate(self, D, Gbar, b):
    gbar_est = D * b + Gbar
    self.write_log('Abs Gbar estimate ' + str(np.dot(gbar_est, gbar_est)))
    return gbar_est