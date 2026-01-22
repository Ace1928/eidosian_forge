import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def check_fit_args_fix(distfn, arg, rvs, method):
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning, message='The shape parameter of the erlang')
        vals = distfn.fit(rvs, floc=0, method=method)
        vals2 = distfn.fit(rvs, fscale=1, method=method)
        npt.assert_(len(vals) == 2 + len(arg))
        npt.assert_(vals[-2] == 0)
        npt.assert_(vals2[-1] == 1)
        npt.assert_(len(vals2) == 2 + len(arg))
        if len(arg) > 0:
            vals3 = distfn.fit(rvs, f0=arg[0], method=method)
            npt.assert_(len(vals3) == 2 + len(arg))
            npt.assert_(vals3[0] == arg[0])
        if len(arg) > 1:
            vals4 = distfn.fit(rvs, f1=arg[1], method=method)
            npt.assert_(len(vals4) == 2 + len(arg))
            npt.assert_(vals4[1] == arg[1])
        if len(arg) > 2:
            vals5 = distfn.fit(rvs, f2=arg[2], method=method)
            npt.assert_(len(vals5) == 2 + len(arg))
            npt.assert_(vals5[2] == arg[2])