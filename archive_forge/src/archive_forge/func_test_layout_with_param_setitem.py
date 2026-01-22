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
def test_layout_with_param_setitem(document, comm):
    import param

    class TestClass(param.Parameterized):
        select = param.ObjectSelector(default=0, objects=[0, 1])

        def __init__(self, **params):
            super().__init__(**params)
            self._layout = Row(Param(self.param, parameters=['select']), self.select)

        @param.depends('select', watch=True)
        def _load(self):
            self._layout[-1] = self.select
    test = TestClass()
    model = test._layout.get_root(document, comm=comm)
    test.select = 1
    assert model.children[1].text == '&lt;pre&gt;1&lt;/pre&gt;'