import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_5_bit_odd_size_no_pad():
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-5ch-9S-5bit.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.uint8))
        assert_equal(data.shape, (9, 5))
        assert_equal(data & 7, 0)
        assert_equal(data.max(), 248)
        assert_equal(data[0, 0], 128)
        assert_equal(data.min(), 0)
        del data