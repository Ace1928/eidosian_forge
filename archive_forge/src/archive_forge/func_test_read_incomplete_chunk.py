import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_incomplete_chunk():
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="Incomplete chunk ID.*b'f'"):
                wavfile.read(fp, mmap=mmap)