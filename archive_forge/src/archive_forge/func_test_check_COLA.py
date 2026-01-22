import sys
import numpy as np
from numpy.testing import (assert_, assert_approx_equal,
import pytest
from pytest import raises as assert_raises
from scipy import signal
from scipy.fft import fftfreq
from scipy.integrate import trapezoid
from scipy.signal import (periodogram, welch, lombscargle, coherence,
from scipy.signal._spectral_py import _spectral_helper
from scipy.signal.tests._scipy_spectral_test_shim import stft_compare as stft
from scipy.signal.tests._scipy_spectral_test_shim import istft_compare as istft
from scipy.signal.tests._scipy_spectral_test_shim import csd_compare as csd
def test_check_COLA(self):
    settings = [('boxcar', 10, 0), ('boxcar', 10, 9), ('bartlett', 51, 26), ('hann', 256, 128), ('hann', 256, 192), ('blackman', 300, 200), (('tukey', 0.5), 256, 64), ('hann', 256, 255)]
    for setting in settings:
        msg = '{}, {}, {}'.format(*setting)
        assert_equal(True, check_COLA(*setting), err_msg=msg)