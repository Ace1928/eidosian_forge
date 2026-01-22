import warnings
import numpy as np
import pytest
from scipy.fft._fftlog import fht, ifht, fhtoffset
from scipy.special import poch
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_close
@array_api_compatible
@pytest.mark.parametrize('n', [64, 63])
def test_fht_exact(n, xp):
    rng = np.random.RandomState(3491349965)
    mu = rng.uniform(0, 3)
    gamma = rng.uniform(-1 - mu, 1 / 2)
    r = np.logspace(-2, 2, n)
    a = xp.asarray(r ** gamma)
    dln = np.log(r[1] / r[0])
    offset = fhtoffset(dln, mu, initial=0.0, bias=gamma)
    A = fht(a, dln, mu, offset=offset, bias=gamma)
    k = np.exp(offset) / r[::-1]
    At = xp.asarray((2 / k) ** gamma * poch((mu + 1 - gamma) / 2, gamma))
    xp_assert_close(A, At)