from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def test_init_functions_are_called():
    init_was_called = {}

    def register_init(name, model, X=None, Y=None):
        init_was_called[name] = True
    layer1 = Linear(5)
    layer2 = Linear(5)
    layer3 = Linear(5)
    layer1.init = partial(register_init, 'one')
    layer2.init = partial(register_init, 'two')
    layer3.init = partial(register_init, 'three')
    model = chain(layer1, chain(layer2, layer3))
    assert not init_was_called
    model.initialize()
    assert init_was_called['one']
    assert init_was_called['two']
    assert init_was_called['three']