import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_clone():
    div1 = Div()
    div2 = Div()
    tabs = Tabs(pn.panel(div1), pn.panel(div2))
    clone = tabs.clone()
    assert [(k, v) for k, v in tabs.param.values().items() if k != 'name'] == [(k, v) for k, v in clone.param.values().items() if k != 'name']