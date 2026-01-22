from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
def is_longdouble_binary_compatible():
    try:
        one = np.frombuffer(b'\x00\x00\x00\x00\x00\x00\x00\x80\xff?\x00\x00\x00\x00\x00\x00', dtype='<f16')
        return one == np.longdouble(1.0)
    except TypeError:
        return False