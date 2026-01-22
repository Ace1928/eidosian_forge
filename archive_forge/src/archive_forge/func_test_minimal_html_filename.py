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
def test_minimal_html_filename(self):
    w = IntText(4)
    tmpd = tempfile.mkdtemp()
    try:
        output = os.path.join(tmpd, 'test.html')
        state = dependency_state(w, drop_defaults=True)
        embed_minimal_html(output, views=w, drop_defaults=True, state=state)
        with open(output, 'r') as f:
            content = f.read()
        assert content.splitlines()[0] == '<!DOCTYPE html>'
    finally:
        shutil.rmtree(tmpd)