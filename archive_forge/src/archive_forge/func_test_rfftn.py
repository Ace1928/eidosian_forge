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
@array_api_compatible
@skip_if_array_api_backend('torch')
def test_rfftn(self, xp):
    x = xp.asarray(random((30, 20, 10)))
    expect = fft.fftn(x)[:, :, :6]
    xp_assert_close(fft.rfftn(x), expect)
    xp_assert_close(fft.rfftn(x, norm='backward'), expect)
    xp_assert_close(fft.rfftn(x, norm='ortho'), expect / xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64)))
    xp_assert_close(fft.rfftn(x, norm='forward'), expect / (30 * 20 * 10))