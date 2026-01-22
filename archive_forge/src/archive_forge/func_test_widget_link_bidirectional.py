import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
def test_widget_link_bidirectional():
    t1 = TextInput()
    t2 = TextInput()
    t1.link(t2, value='value', bidirectional=True)
    t1.value = 'ABC'
    assert t1.value == 'ABC'
    assert t2.value == 'ABC'
    t2.value = 'DEF'
    assert t1.value == 'DEF'
    assert t2.value == 'DEF'