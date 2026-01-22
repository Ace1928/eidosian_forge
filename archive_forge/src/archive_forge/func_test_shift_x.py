import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
def test_shift_x(self):
    y = upfirdn([1, 1], [1.0], 1, 1)
    assert_allclose(y, [1, 1])
    y = upfirdn([1, 1], [0.0, 1.0], 1, 1)
    assert_allclose(y, [0, 1, 1])