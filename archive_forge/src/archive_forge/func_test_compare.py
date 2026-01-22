import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_compare(self):
    taps = firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], weight=[1, 2])
    known_taps = [-0.000626930101730182, -0.103354450635036, -0.00981576747564301, 0.317271686090449, 0.511409425599933, 0.317271686090449, -0.00981576747564301, -0.103354450635036, -0.000626930101730182]
    assert_allclose(taps, known_taps)
    taps = firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], weight=[1, 2])
    known_taps = [0.058545300496815, -0.014233383714318, -0.104688258464392, 0.012403323025279, 0.317930861136062, 0.4880472200297, 0.317930861136062, 0.012403323025279, -0.104688258464392, -0.014233383714318, 0.058545300496815]
    assert_allclose(taps, known_taps)
    taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], fs=20)
    known_taps = [1.156090832768218, -4.138589472739585, 7.528861916432183, -8.553057259294786, 7.528861916432183, -4.138589472739585, 1.156090832768218]
    assert_allclose(taps, known_taps)
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
        taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], nyq=10)
        assert_allclose(taps, known_taps)
        with pytest.raises(ValueError, match='between 0 and 1'):
            firls(7, [0, 1], [0, 1], nyq=0.5)