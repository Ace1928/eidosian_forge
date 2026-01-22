import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
def calculate_maximum_inconsistencies(Z, R, k=3, xp=np):
    n = Z.shape[0] + 1
    dtype = xp.result_type(Z, R)
    B = xp.zeros((n - 1,), dtype=dtype)
    q = xp.zeros((3,))
    for i in range(0, n - 1):
        q[:] = 0.0
        left = Z[i, 0]
        right = Z[i, 1]
        if left >= n:
            q[0] = B[xp.asarray(left, dtype=xp.int64) - n]
        if right >= n:
            q[1] = B[xp.asarray(right, dtype=xp.int64) - n]
        q[2] = R[i, k]
        B[i] = xp.max(q)
    return B