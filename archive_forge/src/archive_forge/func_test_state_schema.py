from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_state_schema():
    from ipywidgets.widgets import IntSlider, Widget
    import json
    import jsonschema
    s = IntSlider()
    state = Widget.get_manager_state(drop_defaults=True)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../', 'state.schema.json')) as f:
        schema = json.load(f)
    jsonschema.validate(state, schema)