import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
@pytest.fixture(params=[1, 2, 9])
def nB(request):
    return request.param