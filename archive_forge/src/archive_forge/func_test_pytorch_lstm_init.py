import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
@pytest.mark.skipif(not has_torch, reason='needs PyTorch')
def test_pytorch_lstm_init():
    model = with_padded(PyTorchLSTM(2, 2, depth=0)).initialize()
    assert model.name == 'with_padded(noop)'