from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_interact_options():

    def f(x):
        return x
    w = interact.options(manual=False).options(manual=True)(f, x=21).widget
    assert w.manual == True
    w = interact_manual.options(manual=False).options()(x=21).widget(f)
    assert w.manual == False
    w = interact(x=21)().options(manual=True)(f).widget
    assert w.manual == True