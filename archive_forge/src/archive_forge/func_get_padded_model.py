import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def get_padded_model():

    def _trim_padded_forward(model, Xp, is_train):

        def backprop(dYp):
            dY = dYp.data
            dX = model.ops.alloc3f(dY.shape[0], dY.shape[1], dY.shape[2] + 1)
            return Padded(dX, dYp.size_at_t, dYp.lengths, dYp.indices)
        assert isinstance(Xp, Padded)
        X = Xp.data
        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        X = X[:, :-1]
        X = X.reshape((Xp.data.shape[0], Xp.data.shape[1], X.shape[1]))
        return (Padded(X, Xp.size_at_t, Xp.lengths, Xp.indices), backprop)
    return with_padded(Model('trimpadded', _trim_padded_forward))