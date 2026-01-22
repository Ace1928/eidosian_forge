import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_unsupported_mmap():
    for filename in {'test-8000Hz-le-3ch-5S-24bit.wav', 'test-8000Hz-le-3ch-5S-36bit.wav', 'test-8000Hz-le-3ch-5S-45bit.wav', 'test-8000Hz-le-3ch-5S-53bit.wav', 'test-8000Hz-le-1ch-10S-20bit-extra.wav'}:
        with raises(ValueError, match='mmap.*not compatible'):
            rate, data = wavfile.read(datafile(filename), mmap=True)