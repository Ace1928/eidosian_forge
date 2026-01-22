import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
def upfirdn_naive(x, h, up=1, down=1):
    """Naive upfirdn processing in Python.

    Note: arg order (x, h) differs to facilitate apply_along_axis use.
    """
    h = np.asarray(h)
    out = np.zeros(len(x) * up, x.dtype)
    out[::up] = x
    out = np.convolve(h, out)[::down][:_output_len(len(h), len(x), up, down)]
    return out