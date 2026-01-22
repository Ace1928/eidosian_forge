import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
@pytest.fixture
def model1(nH, nI):
    return Linear(nH, nI)