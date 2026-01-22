import pytest
import srsly
from thinc.api import (
def test_pickle_with_flatten(linear):
    Xs = [linear.ops.alloc2f(2, 3), linear.ops.alloc2f(4, 3)]
    model = with_array(linear).initialize()
    pickled = srsly.pickle_dumps(model)
    loaded = srsly.pickle_loads(pickled)
    Ys = loaded.predict(Xs)
    assert len(Ys) == 2
    assert Ys[0].shape == (Xs[0].shape[0], linear.get_dim('nO'))
    assert Ys[1].shape == (Xs[1].shape[0], linear.get_dim('nO'))