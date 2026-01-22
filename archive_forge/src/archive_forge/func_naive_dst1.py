from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
def naive_dst1(x, norm=None):
    """Calculate textbook definition version of DST-I."""
    x = np.array(x, copy=True)
    N = len(x)
    M = N + 1
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += 2 * x[n] * np.sin(np.pi * (n + 1.0) * (k + 1.0) / M)
    if norm == 'ortho':
        y *= np.sqrt(0.5 / M)
    return y