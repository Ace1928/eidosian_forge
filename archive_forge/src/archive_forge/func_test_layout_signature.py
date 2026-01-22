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
@pytest.mark.parametrize('panel', all_panels)
def test_layout_signature(panel):
    from inspect import signature
    parameters = signature(panel).parameters
    assert len(parameters) == 2, 'Found following parameters %r on %s' % (parameters, panel)
    assert 'objects' in parameters