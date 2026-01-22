import pytest
import srsly
from thinc.api import (
def test_serialize_refs_roundtrip_bytes():
    fwd = lambda model, X, is_train: (X, lambda dY: dY)
    model_a = Model('a', fwd)
    model = Model('test', fwd, refs={'a': model_a, 'b': None}).initialize()
    with pytest.raises(ValueError):
        model.to_bytes()
    model = Model('test', fwd, refs={'a': model_a, 'b': None}, layers=[model_a])
    assert model.ref_names == ('a', 'b')
    model_bytes = model.to_bytes()
    with pytest.raises(ValueError):
        Model('test', fwd).from_bytes(model_bytes)
    new_model = Model('test', fwd, layers=[model_a])
    new_model.from_bytes(model_bytes)
    assert new_model.ref_names == ('a', 'b')