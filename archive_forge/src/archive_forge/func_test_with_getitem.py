import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def test_with_getitem():
    data = (numpy.asarray([[1, 2, 3, 4]], dtype='f'), numpy.asarray([[5, 6, 7, 8]], dtype='f'))
    model = with_getitem(1, Linear())
    model.initialize(data, data)
    Y, backprop = model.begin_update(data)
    assert len(Y) == len(data)
    assert numpy.array_equal(Y[0], data[0])
    assert not numpy.array_equal(Y[1], data[1])
    dX = backprop(Y)
    assert numpy.array_equal(dX[0], data[0])
    assert not numpy.array_equal(dX[1], data[1])