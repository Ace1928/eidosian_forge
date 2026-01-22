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
@pytest.mark.parametrize('layout', [Row, Column, FlexBox])
def test_pass_objects_ref(document, comm, layout):
    multi_select = MultiSelect(options=['foo', 'bar', 'baz'], value=['bar', 'baz'])
    col = layout(objects=multi_select)
    col.get_root(document, comm=comm)
    assert len(col.objects) == 2
    md1, md2 = col.objects
    assert isinstance(md1, Markdown)
    assert md1.object == 'bar'
    assert isinstance(md2, Markdown)
    assert md2.object == 'baz'
    multi_select.value = ['foo']
    assert len(col.objects) == 1
    md3 = col.objects[0]
    assert isinstance(md3, Markdown)
    assert md3.object == 'foo'