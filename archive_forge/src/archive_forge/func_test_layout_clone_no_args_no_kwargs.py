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
def test_layout_clone_no_args_no_kwargs(panel):
    div1 = Div()
    div2 = Div()
    layout = panel(div1, div2, width=400, sizing_mode='stretch_height')
    clone = layout.clone()
    assert layout.objects[0].object is clone.objects[0].object
    assert layout.objects[1].object is clone.objects[1].object
    assert clone.width == 400
    assert clone.sizing_mode == 'stretch_height'