import numpy
import pytest
from hypothesis import given, settings
from mock import MagicMock
from numpy.testing import assert_allclose
from thinc.api import SGD, Dropout, Linear, chain
from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape
def test_predict_bias(model2):
    input_ = model2.ops.alloc2f(1, model2.get_dim('nI'))
    target_scores = model2.ops.alloc2f(1, model2.get_dim('nI'))
    scores = model2.predict(input_)
    assert_allclose(scores[0], target_scores[0])
    model2.get_param('b')[0] = 2.0
    target_scores[0, 0] = 2.0
    scores = model2.predict(input_)
    assert_allclose(scores, target_scores)
    model2.get_param('b')[1] = 5.0
    target_scores[0, 1] = 5.0
    scores = model2.predict(input_)
    assert_allclose(scores, target_scores)