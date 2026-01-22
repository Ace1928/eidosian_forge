from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask import config
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq
@pytest.mark.parametrize('xp', [np, da])
@pytest.mark.parametrize('N, M, k, dtype, chunks', [(3, None, 0, float, 'auto'), (4, None, 0, float, 'auto'), (3, 4, 0, bool, 'auto'), (3, None, 1, int, 'auto'), (3, None, -1, int, 'auto'), (3, None, 2, int, 1), (6, 8, -2, int, (3, 4)), (6, 8, 0, int, (3, 'auto'))])
def test_tri_like(xp, N, M, k, dtype, chunks):
    args = [N, M, k, dtype]
    cp_a = cupy.tri(*args)
    if xp is da:
        args.append(chunks)
    xp_a = xp.tri(*args, like=da.from_array(cupy.array(())))
    assert_eq(xp_a, cp_a)