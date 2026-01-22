from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
def ref_2d(func, x, **kwargs):
    """Calculate 2-D reference data from a 1d transform"""
    x = np.array(x, copy=True)
    for row in range(x.shape[0]):
        x[row, :] = func(x[row, :], **kwargs)
    for col in range(x.shape[1]):
        x[:, col] = func(x[:, col], **kwargs)
    return x