import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_int_to_float(echo):
    w = NumberWidget()
    with pytest.raises(TraitError):
        w.set_state(dict(i=3.5))