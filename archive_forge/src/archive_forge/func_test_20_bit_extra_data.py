import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_20_bit_extra_data():
    filename = 'test-8000Hz-le-1ch-10S-20bit-extra.wav'
    rate, data = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 1234)
    assert_(np.issubdtype(data.dtype, np.int32))
    assert_equal(data.shape, (10,))
    assert_equal(data & 255, 0)
    assert_((data & 3840).any())
    assert_equal(data, [+2147479552, -2147479552, +2147479552 >> 1, -2147479552 >> 1, +2147479552 >> 2, -2147479552 >> 2, +2147479552 >> 3, -2147479552 >> 3, +2147479552 >> 4, -2147479552 >> 4])