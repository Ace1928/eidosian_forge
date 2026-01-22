from scipy import fft
import numpy as np
import pytest
from numpy.testing import assert_allclose
import multiprocessing
import os
def test_invalid_workers(x):
    cpus = os.cpu_count()
    fft.ifft([1], workers=-cpus)
    with pytest.raises(ValueError, match='workers must not be zero'):
        fft.fft(x, workers=0)
    with pytest.raises(ValueError, match='workers value out of range'):
        fft.ifft(x, workers=-cpus - 1)