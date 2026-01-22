import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_append(document, comm, tabs):
    model = tabs.get_root(document, comm=comm)
    tab1_before, tab2_before = model.tabs
    div3 = Div()
    tabs.append(div3)
    tab1, tab2, tab3 = model.tabs
    assert_tab_is_similar(tab1_before, tab1)
    assert_tab_is_similar(tab2_before, tab2)
    assert tab3.child is div3