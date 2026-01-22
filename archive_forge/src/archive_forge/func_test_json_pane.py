import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_json_pane(document, comm):
    pane = JSON({'a': 2})
    model = pane.get_root(document, comm=comm)
    assert model.text == '{"a": 2}'
    assert pane._models[model.ref['id']][0] is model
    pane.object = '{"b": 3}'
    assert model.text == '{"b": 3}'
    assert pane._models[model.ref['id']][0] is model
    pane.object = {'test': "can't show this"}
    assert model.text == '{"test": "can\'t show this"}'
    assert pane._models[model.ref['id']][0] is model
    pane.object = ["can't show this"]
    assert model.text == '["can\'t show this"]'
    assert pane._models[model.ref['id']][0] is model
    pane.object = "can't show this"
    assert model.text == '"can\'t show this"'
    assert pane._models[model.ref['id']][0] is model
    pane.object = 'can show this'
    assert model.text == '"can show this"'
    assert pane._models[model.ref['id']][0] is model
    pane._cleanup(model)
    assert pane._models == {}