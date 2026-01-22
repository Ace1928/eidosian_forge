import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_numbers_int(echo):
    w = NumberWidget()
    w.set_state(dict(f=1, cf=2, i=3, ci=4))
    assert len(w.comm.messages) == (1 if echo else 0)