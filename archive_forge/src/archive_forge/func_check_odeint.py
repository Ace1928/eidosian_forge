import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.integrate import odeint
import scipy.integrate._test_odeint_banded as banded5x5
def check_odeint(jactype):
    if jactype == JACTYPE_FULL:
        ml = None
        mu = None
        jacobian = jac
    elif jactype == JACTYPE_BANDED:
        ml = 2
        mu = 1
        jacobian = bjac
    else:
        raise ValueError(f'invalid jactype: {jactype!r}')
    y0 = np.arange(1.0, 6.0)
    rtol = 1e-11
    atol = 1e-13
    dt = 0.125
    nsteps = 64
    t = dt * np.arange(nsteps + 1)
    sol, info = odeint(rhs, y0, t, Dfun=jacobian, ml=ml, mu=mu, atol=atol, rtol=rtol, full_output=True)
    yfinal = sol[-1]
    odeint_nst = info['nst'][-1]
    odeint_nfe = info['nfe'][-1]
    odeint_nje = info['nje'][-1]
    y1 = y0.copy()
    nst, nfe, nje = banded5x5.banded5x5_solve(y1, nsteps, dt, jactype)
    assert_allclose(yfinal, y1, rtol=1e-12)
    assert_equal((odeint_nst, odeint_nfe, odeint_nje), (nst, nfe, nje))