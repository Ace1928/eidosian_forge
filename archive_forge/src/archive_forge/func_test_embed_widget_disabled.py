import glob
import json
import os
from io import StringIO
import pytest
from bokeh.models import CustomJS
from panel import Row
from panel.config import config
from panel.io.embed import embed_state
from panel.pane import Str
from panel.param import Param
from panel.widgets import (
def test_embed_widget_disabled(document, comm):
    select = Select(options=['A', 'B', 'C'], disabled=True)
    string = Str()
    select.link(string, value='object')
    string.param.watch(print, 'object')
    panel = Row(select, string)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    assert embed_state(panel, model, document) is None