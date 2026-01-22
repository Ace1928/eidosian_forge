from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def test_infer_output_shape():
    model = Relu(dropout=0.2)
    X = model.ops.alloc2f(4, 5)
    Y = model.ops.alloc2f(4, 2)
    assert model.has_dim('nI') is None
    assert model.has_dim('nO') is None
    model.initialize(X=X, Y=Y)
    assert model.get_dim('nI') == 5
    assert model.get_dim('nO') == 2