import pytest
import srsly
from thinc.api import (
def test_serialize_model_shims_roundtrip_bytes():
    fwd = lambda model, X, is_train: (X, lambda dY: dY)
    test_shim = SerializableShim(None)
    shim_model = Model('shimmodel', fwd, shims=[test_shim])
    model = chain(Linear(2, 3), shim_model, Maxout(2, 3))
    model.initialize()
    assert model.layers[1].shims[0].value == 'shimdata'
    model_bytes = model.to_bytes()
    with pytest.raises(ValueError):
        Linear(2, 3).from_bytes(model_bytes)
    test_shim = SerializableShim(None)
    shim_model = Model('shimmodel', fwd, shims=[test_shim])
    new_model = chain(Linear(2, 3), shim_model, Maxout(2, 3)).from_bytes(model_bytes)
    assert new_model.layers[1].shims[0].value == 'shimdata from bytes'