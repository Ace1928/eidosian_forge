import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
@pytest.mark.parametrize('dtype', [np.float16, np.longdouble])
def test_dtypes_nonstandard(self, dtype):
    x = random(30).astype(dtype)
    out_dtypes = {np.float16: np.complex64, np.longdouble: np.clongdouble}
    x_complex = x.astype(out_dtypes[dtype])
    res_fft = fft.ifft(fft.fft(x))
    res_rfft = fft.irfft(fft.rfft(x))
    res_hfft = fft.hfft(fft.ihfft(x), x.shape[0])
    assert_array_almost_equal(res_fft, x_complex)
    assert_array_almost_equal(res_rfft, x)
    assert_array_almost_equal(res_hfft, x)
    assert res_fft.dtype == x_complex.dtype
    assert res_rfft.dtype == np.result_type(np.float32, x.dtype)
    assert res_hfft.dtype == np.result_type(np.float32, x.dtype)