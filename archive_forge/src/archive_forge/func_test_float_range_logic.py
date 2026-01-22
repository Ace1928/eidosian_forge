from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_float_range_logic():
    frsw = widgets.FloatRangeSlider
    w = frsw(value=(0.2, 0.4), min=0.0, max=0.6)
    check_widget(w, cls=frsw, value=(0.2, 0.4), min=0.0, max=0.6)
    w.min = 0.0
    w.max = 0.6
    w.lower = 0.2
    w.upper = 0.4
    check_widget(w, cls=frsw, value=(0.2, 0.4), min=0.0, max=0.6)
    w.value = (0.0, 0.1)
    check_widget(w, cls=frsw, value=(0.0, 0.1), min=0.0, max=0.6)
    w.value = (0.5, 0.6)
    check_widget(w, cls=frsw, value=(0.5, 0.6), min=0.0, max=0.6)
    w.lower = 0.2
    check_widget(w, cls=frsw, value=(0.2, 0.6), min=0.0, max=0.6)
    with pytest.raises(TraitError):
        w.min = 0.7
    with pytest.raises(TraitError):
        w.max = -0.1
    w = frsw(min=2, max=3, value=(2.2, 2.5))
    check_widget(w, min=2.0, max=3.0)
    with pytest.raises(TraitError):
        frsw(min=0.2, max=0.1)