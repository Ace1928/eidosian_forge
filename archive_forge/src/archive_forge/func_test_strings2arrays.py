import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
def test_strings2arrays():
    strings = ['hello', 'world']
    model = strings2arrays()
    Y, backprop = model.begin_update(strings)
    assert len(Y) == len(strings)
    assert backprop([]) == []