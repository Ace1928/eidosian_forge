from __future__ import annotations
import contextlib
from itertools import combinations_with_replacement
import numpy as np
import pytest
import dask.array as da
import dask.array.fft
from dask.array.core import normalize_chunks
from dask.array.fft import fft_wrap
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('n', [1, 2, 3, 6, 7])
@pytest.mark.parametrize('d', [1.0, 0.5, 2 * np.pi])
@pytest.mark.parametrize('c', [lambda m: (m // 2 + 1,), lambda m: (1, m // 2)])
def test_rfftfreq(n, d, c):
    c = [ci for ci in c(n) if ci != 0]
    r1 = np.fft.rfftfreq(n, d)
    r2 = da.fft.rfftfreq(n, d, chunks=c)
    assert normalize_chunks(c, r2.shape) == r2.chunks
    assert_eq(r1, r2)