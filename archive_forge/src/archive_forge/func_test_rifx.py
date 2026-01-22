import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_rifx():
    for rifx, riff in {('test-44100Hz-be-1ch-4bytes.wav', 'test-44100Hz-le-1ch-4bytes.wav'), ('test-8000Hz-be-3ch-5S-24bit.wav', 'test-8000Hz-le-3ch-5S-24bit.wav')}:
        rate1, data1 = wavfile.read(datafile(rifx), mmap=False)
        rate2, data2 = wavfile.read(datafile(riff), mmap=False)
        assert_equal(rate1, rate2)
        assert_equal(data1, data2)