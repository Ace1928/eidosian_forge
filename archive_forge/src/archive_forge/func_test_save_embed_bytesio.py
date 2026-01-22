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
def test_save_embed_bytesio():
    checkbox = Checkbox()
    string = Str()

    def link(target, event):
        target.object = event.new
    checkbox.link(string, callbacks={'value': link})
    panel = Row(checkbox, string)
    stringio = StringIO()
    panel.save(stringio, embed=True)
    stringio.seek(0)
    utf = stringio.read()
    assert '&amp;lt;pre&amp;gt;False&amp;lt;/pre&amp;gt;' in utf
    assert '&amp;lt;pre&amp;gt;True&amp;lt;/pre&amp;gt;' in utf