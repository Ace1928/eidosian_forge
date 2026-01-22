from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
def naive_dct4(x, norm=None):
    """Calculate textbook definition version of DCT-IV."""
    x = np.array(x, copy=True)
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n] * np.cos(np.pi * (n + 0.5) * (k + 0.5) / N)
    if norm == 'ortho':
        y *= np.sqrt(2.0 / N)
    else:
        y *= 2
    return y