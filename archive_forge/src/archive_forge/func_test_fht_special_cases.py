import warnings
import numpy as np
import pytest
from scipy.fft._fftlog import fht, ifht, fhtoffset
from scipy.special import poch
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_close
@array_api_compatible
def test_fht_special_cases(xp):
    rng = np.random.RandomState(3491349965)
    a = xp.asarray(rng.standard_normal(64))
    dln = rng.uniform(-1, 1)
    mu, bias = (-4.0, 1.0)
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias)
        assert not record, 'fht warned about a well-defined transform'
    mu, bias = (-2.5, 0.5)
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias)
        assert not record, 'fht warned about a well-defined transform'
    mu, bias = (-3.5, 0.5)
    with pytest.warns(Warning) as record:
        fht(a, dln, mu, bias=bias)
        assert record, 'fht did not warn about a singular transform'
    mu, bias = (-2.5, 0.5)
    with pytest.warns(Warning) as record:
        ifht(a, dln, mu, bias=bias)
        assert record, 'ifht did not warn about a singular transform'