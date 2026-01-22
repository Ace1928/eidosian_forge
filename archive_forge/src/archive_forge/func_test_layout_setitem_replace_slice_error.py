import param
import pytest
from bokeh.models import Column as BkColumn, Div, Row as BkRow
from panel.chat import ChatInterface
from panel.layout import (
from panel.layout.base import ListPanel, NamedListPanel
from panel.pane import Bokeh, Markdown
from panel.param import Param
from panel.tests.util import check_layoutable_properties
from panel.widgets import Debugger, MultiSelect
@pytest.mark.parametrize('panel', [Column, Row])
def test_layout_setitem_replace_slice_error(panel, document, comm):
    div1 = Div()
    div2 = Div()
    div3 = Div()
    layout = panel(div1, div2, div3)
    layout.get_root(document, comm=comm)
    div3 = Div()
    with pytest.raises(IndexError):
        layout[1:] = [div3]