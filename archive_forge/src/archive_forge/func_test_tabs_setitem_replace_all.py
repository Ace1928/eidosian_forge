import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_setitem_replace_all(document, comm):
    div1 = Div()
    div2 = Div()
    layout = Tabs(div1, div2)
    p1, p2 = layout.objects
    model = layout.get_root(document, comm=comm)
    assert p1._models[model.ref['id']][0] is model.tabs[0].child
    div3 = Div()
    div4 = Div()
    layout[:] = [('B', div3), ('C', div4)]
    tab1, tab2 = model.tabs
    assert tab1.child is div3
    assert tab1.title == 'B'
    assert tab2.child is div4
    assert tab2.title == 'C'
    assert p1._models == {}
    assert p2._models == {}