import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_recursive_double_wrap():

    def dummy_model(name, layers):
        return Model(name, lambda model, X, is_train: ..., layers=layers)
    relu = Relu(5)
    chained = chain(relu, relu)
    concat = concatenate(chained, chained, relu)
    concat_wrapped = wrap_model_recursive(concat, lambda model: dummy_model(f'dummy({model.name})', [model]))
    n_debug = 0
    for model in concat_wrapped.walk():
        if model.name.startswith('dummy'):
            n_debug += 1
    assert n_debug == 3
    assert concat_wrapped.layers[0].layers[0].layers[0].layers[0].name == 'dummy(relu)'
    assert concat_wrapped.layers[0].layers[0].layers[0].layers[1].name == 'dummy(relu)'
    assert concat_wrapped.layers[0].layers[1].layers[0].layers[0].name == 'dummy(relu)'
    assert concat_wrapped.layers[0].layers[1].layers[0].layers[1].name == 'dummy(relu)'
    assert concat_wrapped.layers[0].layers[2].name == 'dummy(relu)'