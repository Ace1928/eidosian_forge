from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def min_polynomial(self, prec=100, degree=10):
    if self._min_poly is None:
        self_prec = self(prec)
        p = best_algdep_factor(self_prec, degree)
        z = self(2 * prec)
        if acceptable_error(p, z, ZZ(0), 0.2):
            self._min_poly = p
            self._default_precision = prec
            self._approx_root = self_prec
    return self._min_poly