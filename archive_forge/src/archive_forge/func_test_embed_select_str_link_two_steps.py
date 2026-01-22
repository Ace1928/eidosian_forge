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
def test_embed_select_str_link_two_steps(document, comm):
    select = Select(options=['A', 'B', 'C'])
    string1 = Str()
    select.link(string1, value='object')
    string2 = Str()
    string1.link(string2, object='object')
    panel = Row(select, string1, string2)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    embed_state(panel, model, document)
    _, state = document.roots
    assert set(state.state) == {'A', 'B', 'C'}
    for k, v in state.state.items():
        content = json.loads(v['content'])
        assert 'events' in content
        events = content['events']
        assert len(events) == 2
        event = events[0]
        assert event['kind'] == 'ModelChanged'
        assert event['attr'] == 'text'
        assert event['model'] == model.children[1].ref
        assert event['new'] == '&lt;pre&gt;%s&lt;/pre&gt;' % k
        event = events[1]
        assert event['kind'] == 'ModelChanged'
        assert event['attr'] == 'text'
        assert event['model'] == model.children[2].ref
        assert event['new'] == '&lt;pre&gt;%s&lt;/pre&gt;' % k