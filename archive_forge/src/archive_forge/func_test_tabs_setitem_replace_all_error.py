import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_setitem_replace_all_error(document, comm):
    div1 = Div()
    div2 = Div()
    layout = Tabs(div1, div2)
    layout.get_root(document, comm=comm)
    div3 = Div()
    with pytest.raises(IndexError):
        layout[:] = div3