import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_36_bit_odd_size():
    filename = 'test-8000Hz-le-3ch-5S-36bit.wav'
    rate, data = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 8000)
    assert_(np.issubdtype(data.dtype, np.int64))
    assert_equal(data.shape, (5, 3))
    assert_equal(data & 268435455, 0)
    correct = [[-9223372036854775808, -9223372036586340352, -536870912], [-4611686018427387904, -4611686018158952448, -268435456], [+0, +0, +0], [+4611686018427387904, +4611686018158952448, +268435456], [+9223372036586340352, +9223372036586340352, +536870912]]
    assert_equal(data, correct)