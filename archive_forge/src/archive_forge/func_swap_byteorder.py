from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def swap_byteorder(arr):
    """Returns the same array with swapped byteorder"""
    dtype = arr.dtype.newbyteorder('S')
    return arr.astype(dtype)