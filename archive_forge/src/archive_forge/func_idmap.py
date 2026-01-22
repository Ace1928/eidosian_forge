import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def idmap(self, *args):
    if self.spfunc_first:
        res = self.spfunc(*args)
        if np.isnan(res):
            return np.nan
        args = list(args)
        args[self.index] = res
        with mpmath.workdps(self.dps):
            res = self.mpfunc(*tuple(args))
            res = mpf2float(res.real)
    else:
        with mpmath.workdps(self.dps):
            res = self.mpfunc(*args)
            res = mpf2float(res.real)
        args = list(args)
        args[self.index] = res
        res = self.spfunc(*tuple(args))
    return res