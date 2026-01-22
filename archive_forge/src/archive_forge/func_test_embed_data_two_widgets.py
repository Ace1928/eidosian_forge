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
def test_embed_data_two_widgets(self):
    w1 = IntText(4)
    w2 = IntSlider(min=0, max=100)
    jslink((w1, 'value'), (w2, 'value'))
    state = dependency_state([w1, w2], drop_defaults=True)
    data = embed_data(views=[w1, w2], drop_defaults=True, state=state)
    state = data['manager_state']['state']
    views = data['view_specs']
    assert len(state) == 7
    assert len(views) == 2
    model_names = [s['model_name'] for s in state.values()]
    assert 'IntTextModel' in model_names
    assert 'IntSliderModel' in model_names