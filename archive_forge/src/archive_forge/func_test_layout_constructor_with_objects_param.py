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
@pytest.mark.parametrize('panel', [Card, Column, Row])
def test_layout_constructor_with_objects_param(panel):
    div1 = Div()
    div2 = Div()
    layout = panel(objects=[div1, div2])
    assert all((isinstance(p, Bokeh) for p in layout.objects))