import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
def test_widget_jslink_bidirectional(document, comm):
    t1 = TextInput()
    t2 = TextInput()
    t1.jslink(t2, value='value', bidirectional=True)
    row = Row(t1, t2)
    model = row.get_root(document, comm)
    tm1, tm2 = model.children
    link1_customjs = tm1.js_property_callbacks['change:value'][-1]
    link2_customjs = tm2.js_property_callbacks['change:value'][-1]
    assert link1_customjs.args['source'] is tm1
    assert link2_customjs.args['source'] is tm2
    assert link1_customjs.args['target'] is tm2
    assert link2_customjs.args['target'] is tm1