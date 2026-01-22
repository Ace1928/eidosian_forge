from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_line_search_wolfe1(self):
    c = 0
    smax = 100
    for name, f, fprime, x, p, old_f in self.line_iter():
        f0 = f(x)
        g0 = fprime(x)
        self.fcount = 0
        s, fc, gc, fv, ofv, gv = ls.line_search_wolfe1(f, fprime, x, p, g0, f0, old_f, amax=smax)
        assert_equal(self.fcount, fc + gc)
        assert_fp_equal(ofv, f(x))
        if s is None:
            continue
        assert_fp_equal(fv, f(x + s * p))
        assert_array_almost_equal(gv, fprime(x + s * p), decimal=14)
        if s < smax:
            c += 1
            assert_line_wolfe(x, p, s, f, fprime, err_msg=name)
    assert c > 3