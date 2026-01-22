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
def test_pass_bind_function_by_reference():
    int_input = IntInput(start=0, end=400, value=42)
    fn = bind(lambda v: v + 10, int_input)
    text_input = TextInput(width=fn)
    assert text_input.width == 52
    int_input.value = 101
    assert text_input.width == 111