from scipy import fft
import numpy as np
import pytest
from numpy.testing import assert_allclose
import multiprocessing
import os
def test_mixed_threads_processes(x):
    expect = fft.fft(x, workers=2)
    with multiprocessing.Pool(2) as p:
        res = p.map(_mt_fft, [x for _ in range(4)])
    for r in res:
        assert_allclose(r, expect)
    fft.fft(x, workers=2)