from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
def fftw_dst_ref(type, size, dt, reference_data):
    x = np.linspace(0, size - 1, size).astype(dt)
    dt = np.result_type(np.float32, dt)
    if dt == np.float64:
        data = reference_data['FFTWDATA_DOUBLE']
    elif dt == np.float32:
        data = reference_data['FFTWDATA_SINGLE']
    elif dt == np.longdouble:
        data = reference_data['FFTWDATA_LONGDOUBLE']
    else:
        raise ValueError()
    y = data['dst_%d_%d' % (type, size)].astype(dt)
    return (x, y, dt)