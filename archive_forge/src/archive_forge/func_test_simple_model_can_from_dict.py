import pytest
import srsly
from thinc.api import (
def test_simple_model_can_from_dict():
    model = Maxout(5, 10, nP=2).initialize()
    model_dict = model.to_dict()
    assert model.can_from_dict(model_dict)
    assert Maxout(5, 10, nP=2).can_from_dict(model_dict)
    assert not Maxout(10, 5, nP=2).can_from_dict(model_dict)
    assert Maxout(5, nP=2).can_from_dict(model_dict)