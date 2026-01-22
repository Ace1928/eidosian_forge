import pytest
import srsly
from thinc.api import (
def test_multi_model_can_from_dict():
    model = chain(Maxout(5, 10, nP=2), Maxout(2, 3)).initialize()
    model_dict = model.to_dict()
    assert model.can_from_dict(model_dict)
    assert chain(Maxout(5, 10, nP=2), Maxout(2, 3)).can_from_dict(model_dict)
    resized = chain(Maxout(5, 10, nP=3), Maxout(2, 3))
    assert not resized.can_from_dict(model_dict)