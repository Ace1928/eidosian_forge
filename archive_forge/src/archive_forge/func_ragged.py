import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.types import Pairs, Ragged
@pytest.fixture
def ragged():
    data = numpy.zeros((20, 4), dtype='f')
    lengths = numpy.array([4, 2, 8, 1, 4], dtype='i')
    data[0] = 0
    data[1] = 1
    data[2] = 2
    data[3] = 3
    data[4] = 4
    data[5] = 5
    return Ragged(data, lengths)