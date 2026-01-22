from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_interact_manual_button():
    c = interact.options(manual=True).widget(f)
    w = c.children[0]
    check_widget(w, cls=widgets.Button)