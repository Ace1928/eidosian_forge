from io import StringIO
from html.parser import HTMLParser
import json
import os
import re
import tempfile
import shutil
import traitlets
from ..widgets import IntSlider, IntText, Text, Widget, jslink, HBox, widget_serialization, widget as widget_module
from ..embed import embed_data, embed_snippet, embed_minimal_html, dependency_state
def test_embed_data_complex(self):
    w1 = IntText(4)
    w2 = IntSlider(min=0, max=100)
    jslink((w1, 'value'), (w2, 'value'))
    w3 = CaseWidget()
    w3.a = w1
    w4 = CaseWidget()
    w4.a = w3
    w4.other['test'] = w2
    w3.b = w4
    HBox(children=[w4])
    state = dependency_state(w3)
    assert len(state) == 9
    model_names = [s['model_name'] for s in state.values()]
    assert 'IntTextModel' in model_names
    assert 'IntSliderModel' in model_names
    assert 'CaseWidgetModel' in model_names
    assert 'LinkModel' in model_names
    assert 'HBoxModel' not in model_names
    data = embed_data(views=w3, drop_defaults=True, state=state)
    assert state is data['manager_state']['state']
    views = data['view_specs']
    assert len(views) == 1