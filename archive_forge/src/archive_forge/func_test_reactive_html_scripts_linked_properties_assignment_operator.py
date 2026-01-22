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
def test_reactive_html_scripts_linked_properties_assignment_operator():
    for operator in ['', '+', '-', '*', '\\', '%', '**', '>>', '<<', '>>>', '&', '^', '&&', '||', '??']:
        for sep in [' ', '']:

            class TestScripts(ReactiveHTML):
                clicks = param.Integer()
                _template = "<div id='test'></div>"
                _scripts = {'render': f'test.onclick = () => {{ data.clicks{sep}{operator}= 1 }}'}
            assert TestScripts()._linked_properties == ('clicks',)