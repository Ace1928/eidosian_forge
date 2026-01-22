import numpy
import pytest
from hypothesis import given, settings
from mock import MagicMock
from numpy.testing import assert_allclose
from thinc.api import SGD, Dropout, Linear, chain
from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape
@pytest.mark.parametrize('X,expected', [(numpy.asarray([0.0, 0.0], dtype='f'), [0.0, 0.0]), (numpy.asarray([1.0, 0.0], dtype='f'), [1.0, 0.0]), (numpy.asarray([0.0, 1.0], dtype='f'), [0.0, 1.0]), (numpy.asarray([1.0, 1.0], dtype='f'), [1.0, 1.0])])
def test_predict_weights(X, expected):
    W = numpy.asarray([1.0, 0.0, 0.0, 1.0], dtype='f').reshape((2, 2))
    bias = numpy.asarray([0.0, 0.0], dtype='f')
    model = Linear(W.shape[0], W.shape[1])
    model.set_param('W', W)
    model.set_param('b', bias)
    scores = model.predict(X.reshape((1, -1)))
    assert_allclose(scores.ravel(), expected)