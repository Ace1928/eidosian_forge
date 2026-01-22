import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
@pytest.mark.parametrize('ops', [Ops(), NumpyOps()])
def test_LSTM_fwd_bwd_shapes_simple(ops, nO, nI):
    nO = 1
    nI = 2
    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
    model = with_padded(LSTM(nO, nI)).initialize(X=[X])
    for node in model.walk():
        node.ops = ops
    ys, backprop_ys = model([X], is_train=True)
    dXs = backprop_ys(ys)
    assert numpy.vstack(dXs).shape == numpy.vstack([X]).shape