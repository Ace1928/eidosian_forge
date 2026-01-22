import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_unknown_riff_form_type():
    for mmap in [False, True]:
        filename = 'Transparent Busy.ani'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Not a WAV file.*ACON'):
                wavfile.read(fp, mmap=mmap)