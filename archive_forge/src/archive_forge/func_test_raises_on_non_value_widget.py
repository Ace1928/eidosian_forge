from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_raises_on_non_value_widget():
    """ Test that passing in a non-value widget raises an error """

    class BadWidget(Widget):
        """ A widget that contains a `value` traitlet """
        value = Float()
    with pytest.raises(TypeError, match='.* not a ValueWidget.*'):
        interactive(f, b=BadWidget())