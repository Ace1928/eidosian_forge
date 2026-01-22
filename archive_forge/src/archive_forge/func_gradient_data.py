from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
@pytest.fixture
def gradient_data(nB, nO):
    return numpy.zeros((nB, nO), dtype='f') - 1.0