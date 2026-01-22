from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
def naive_dct1(x, norm=None):
    """Calculate textbook definition version of DCT-I."""
    x = np.array(x, copy=True)
    N = len(x)
    M = N - 1
    y = np.zeros(N)
    m0, m = (1, 2)
    if norm == 'ortho':
        m0 = np.sqrt(1.0 / M)
        m = np.sqrt(2.0 / M)
    for k in range(N):
        for n in range(1, N - 1):
            y[k] += m * x[n] * np.cos(np.pi * n * k / M)
        y[k] += m0 * x[0]
        y[k] += m0 * x[N - 1] * (1 if k % 2 == 0 else -1)
    if norm == 'ortho':
        y[0] *= 1 / np.sqrt(2)
        y[N - 1] *= 1 / np.sqrt(2)
    return y