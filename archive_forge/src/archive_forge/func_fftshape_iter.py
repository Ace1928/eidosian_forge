from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def fftshape_iter(shp):
    if len(shp) <= 0:
        yield ()
    else:
        for j in (shp[0] // 2, shp[0], shp[0] * 2):
            for rest in fftshape_iter(shp[1:]):
                yield ((j,) + rest)