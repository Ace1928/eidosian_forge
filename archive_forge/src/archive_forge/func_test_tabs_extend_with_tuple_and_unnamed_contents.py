import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_extend_with_tuple_and_unnamed_contents(document, comm, tabs):
    model = tabs.get_root(document, comm=comm)
    tab1_before, tab2_before = model.tabs
    div3, div4 = (Div(), Div())
    tabs.extend([('Div4', div4), ('Div3', div3)])
    tab1, tab2, tab3, tab4 = model.tabs
    assert_tab_is_similar(tab1_before, tab1)
    assert_tab_is_similar(tab2_before, tab2)
    assert tab3.child is div4
    assert tab3.title == 'Div4'
    assert tab4.child is div3
    assert tab4.title == 'Div3'