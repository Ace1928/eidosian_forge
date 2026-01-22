from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_list_str():
    values = ['hello', 'there', 'guy']
    first = values[0]
    c = interactive(f, lis=values)
    assert len(c.children) == 2
    d = dict(cls=widgets.Dropdown, value=first, options=tuple(values), _options_labels=tuple(values), _options_values=tuple(values))
    check_widget_children(c, lis=d)