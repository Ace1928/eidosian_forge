import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_implicit_constructor(document, comm):
    div1, div2 = (Div(), Div())
    p1 = pn.panel(div1, name='Div1')
    p2 = pn.panel(div2, name='Div2')
    tabs = Tabs(p1, p2)
    model = tabs.get_root(document, comm=comm)
    assert isinstance(model, BkTabs)
    assert len(model.tabs) == 2
    assert all((isinstance(c, BkPanel) for c in model.tabs))
    tab1, tab2 = model.tabs
    assert tab1.title == tab1.name == p1.name == 'Div1'
    assert tab1.child is div1
    assert tab2.title == tab2.name == p2.name == 'Div2'
    assert tab2.child is div2