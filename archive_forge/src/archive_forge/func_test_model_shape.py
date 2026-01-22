from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def test_model_shape(model, model1, model2, nI, nH, nO):
    assert model.get_dim('nI') == model1.get_dim('nI')
    assert model.get_dim('nO') == model2.get_dim('nO')