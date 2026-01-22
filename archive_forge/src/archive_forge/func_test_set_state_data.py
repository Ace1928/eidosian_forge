import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_data(echo):
    w = DataWidget()
    data = memoryview(b'x' * 30)
    w.set_state(dict(a=True, d={'data': data}))
    assert len(w.comm.messages) == (1 if echo else 0)