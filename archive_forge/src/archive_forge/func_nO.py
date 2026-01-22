import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
@pytest.fixture(params=[1, 2, 7, 9])
def nO(request):
    return request.param