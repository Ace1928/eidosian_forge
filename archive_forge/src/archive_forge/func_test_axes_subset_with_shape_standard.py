import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
@skip_if_array_api_backend('torch')
@array_api_compatible
@pytest.mark.parametrize('op', [fft.fftn, fft.ifftn, fft.rfftn, fft.irfftn])
def test_axes_subset_with_shape_standard(self, op, xp):
    x = xp.asarray(random((16, 8, 4)))
    axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    xp_test = array_namespace(x)
    for a in axes:
        shape = tuple([2 * x.shape[ax] if ax in a[:2] else x.shape[ax] for ax in range(x.ndim)])
        op_tr = op(xp_test.permute_dims(x, axes=a), s=shape[:2], axes=(0, 1))
        tr_op = xp_test.permute_dims(op(x, s=shape[:2], axes=a[:2]), axes=a)
        xp_assert_close(op_tr, tr_op)