import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
def test_output_len_long_input():
    len_h = 1001
    in_len = 10 ** 8
    up = 320
    down = 441
    out_len = _output_len(len_h, in_len, up, down)
    assert out_len == 72562360