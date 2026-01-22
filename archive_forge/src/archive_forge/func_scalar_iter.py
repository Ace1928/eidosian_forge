from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def scalar_iter(self):
    for name, phi, derphi in self.scalar_funcs:
        for old_phi0 in np.random.randn(3):
            yield (name, phi, derphi, old_phi0)