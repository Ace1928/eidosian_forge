import inspect
import platform
from typing import Tuple, cast
import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from packaging.version import Version
from thinc.api import (
from thinc.backends._custom_kernels import KERNELS, KERNELS_LIST, compile_mmh
from thinc.compat import has_cupy_gpu, has_torch, torch_version
from thinc.types import Floats2d
from thinc.util import torch2xp, xp2torch
from .. import strategies
from ..strategies import arrays_BI, ndarrays_of_shape
def get_lstm_args(depth, dirs, nO, batch_size, nI, draw=None):
    if dirs == 1:
        n_params = nO * 4 * nI + nO * 4 + nO * 4 * nO + nO * 4
        for _ in range(1, depth):
            n_params += nO * 4 * nO + nO * 4 + nO * 4 * nO + nO * 4
    else:
        n_params = nO * 2 * nI + nO * 2 + nO * 2 * (nO // 2) + nO * 2
        for _ in range(1, depth):
            n_params += nO * 2 * nO + nO * 2 + nO * 2 * (nO // 2) + nO * 2
        n_params *= 2
    lstm = LSTM(nO, nI, depth=depth, bi=dirs >= 2).initialize()
    assert lstm.get_param('LSTM').size == n_params
    if draw:
        params = draw(ndarrays_of_shape(n_params))
        size_at_t = numpy.ones(shape=(batch_size,), dtype='int32')
        X = draw(ndarrays_of_shape((int(size_at_t.sum()), nI)))
    else:
        params = numpy.ones((n_params,), dtype='f')
        size_at_t = numpy.ones(shape=(batch_size,), dtype='int32')
        X = numpy.zeros((int(size_at_t.sum()), nI))
    H0 = numpy.zeros((depth, dirs, nO // dirs))
    C0 = numpy.zeros((depth, dirs, nO // dirs))
    return (params, H0, C0, X, size_at_t)