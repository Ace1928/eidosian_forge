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
def test_reactive_html_templated_variable_not_in_declared_node():
    with pytest.raises(ValueError) as excinfo:

        class TestTemplatedChildren(ReactiveHTML):
            children = param.List(default=[])
            _template = '\n            <select>\n            {%- for option in children %}\n            <option id="option">{{option   }}</option>\n            {%- endfor %}\n            </select>\n            '
    assert 'could not be expanded because the <select> node' in str(excinfo)
    assert '{%- for option in children %}' in str(excinfo)