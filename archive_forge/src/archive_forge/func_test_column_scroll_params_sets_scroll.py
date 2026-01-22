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
@pytest.mark.parametrize('scroll_param', ['auto_scroll_limit', 'scroll', 'scroll_button_threshold', 'view_latest'])
def test_column_scroll_params_sets_scroll(scroll_param, document, comm):
    if scroll_param not in ['auto_scroll_limit', 'scroll_button_threshold']:
        params = {scroll_param: True}
    else:
        params = {scroll_param: 1}
    col = Column(**params)
    assert getattr(col, scroll_param)
    assert col.scroll