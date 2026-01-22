from functools import partial
import pytest
from thinc.api import Linear, resizable
from thinc.layers.resizable import resize_linear_weighted, resize_model
def test_resizable_linear_default_name(model):
    assert model.name == 'resizable(linear)'