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
def test_layout_setitem_replace_slice(panel, document, comm):
    div1 = Div()
    div2 = Div()
    div3 = Div()
    layout = panel(div1, div2, div3)
    p1, p2, p3 = layout.objects
    model = layout.get_root(document, comm=comm)
    assert p1._models[model.ref['id']][0] is model.children[0]
    div3 = Div()
    div4 = Div()
    layout[1:] = [div3, div4]
    assert model.children == [div1, div3, div4]
    assert p2._models == {}
    assert p3._models == {}