import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_constructor(document, comm):
    div1 = Div()
    div2 = Div()
    tabs = Tabs(('Div1', div1), ('Div2', div2))
    model = tabs.get_root(document, comm=comm)
    assert isinstance(model, BkTabs)
    assert len(model.tabs) == 2
    assert all((isinstance(c, BkPanel) for c in model.tabs))
    tab1, tab2 = model.tabs
    assert tab1.title == 'Div1'
    assert tab1.child is div1
    assert tab2.title == 'Div2'
    assert tab2.child is div2