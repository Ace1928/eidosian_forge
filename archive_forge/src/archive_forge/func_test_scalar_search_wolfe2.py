from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_scalar_search_wolfe2(self):
    for name, phi, derphi, old_phi0 in self.scalar_iter():
        s, phi1, phi0, derphi1 = ls.scalar_search_wolfe2(phi, derphi, phi(0), old_phi0, derphi(0))
        assert_fp_equal(phi0, phi(0), name)
        assert_fp_equal(phi1, phi(s), name)
        if derphi1 is not None:
            assert_fp_equal(derphi1, derphi(s), name)
        assert_wolfe(s, phi, derphi, err_msg=f'{name} {old_phi0:g}')