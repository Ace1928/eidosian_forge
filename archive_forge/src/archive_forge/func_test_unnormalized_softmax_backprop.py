from typing import Tuple, cast
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Model, NumpyOps, Softmax_v2
from thinc.types import Floats2d, Ints1d
from thinc.util import has_torch, torch2xp, xp2torch
def test_unnormalized_softmax_backprop():
    model = Softmax_v2(normalize_outputs=False)
    model.initialize(inputs, outputs)
    _, backprop = model(inputs, is_train=False)
    with pytest.raises(ValueError, match='backprop is not supported'):
        backprop(OPS.xp.zeros_like(outputs))
    _, backprop = model(inputs, is_train=True)
    dX = backprop(OPS.xp.zeros_like(outputs))
    assert OPS.xp.all(dX == 0.0)