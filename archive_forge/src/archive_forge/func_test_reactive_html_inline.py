from __future__ import annotations
import asyncio
import unittest.mock
from functools import partial
from typing import ClassVar, Mapping
import bokeh.core.properties as bp
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from bokeh.models import Div
from panel.depends import bind, depends
from panel.layout import Tabs, WidgetBox
from panel.pane import Markdown
from panel.reactive import Reactive, ReactiveHTML
from panel.viewable import Viewable
from panel.widgets import (
def test_reactive_html_inline():

    class TestInline(ReactiveHTML):
        int = param.Integer(default=3, doc='An integer')
        _template = '<div id="div" onchange=${_div_change} width=${int}></div>'

        def _div_change(self, event):
            pass
    data_model = TestInline._data_model
    assert data_model.__name__ == 'TestInline1'
    properties = data_model.properties()
    assert 'int' in properties
    int_prop = data_model.lookup('int')
    assert isinstance(int_prop.property, bp.Int)
    assert int_prop.class_default(data_model) == 3
    assert TestInline._node_callbacks == {'div': [('onchange', '_div_change')]}
    assert TestInline._inline_callbacks == [('div', 'onchange', '_div_change')]
    test = TestInline()
    root = test.get_root()
    assert test._attrs == {'div': [('onchange', [], '{_div_change}'), ('width', ['int'], '{int}')]}
    assert root.callbacks == {'div': [('onchange', '_div_change')]}
    assert root.events == {}
    test.on_event('div', 'click', print)
    assert root.events == {'div': {'click': False}}