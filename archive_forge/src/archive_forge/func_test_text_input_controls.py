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
def test_text_input_controls():
    text_input = TextInput()
    controls = text_input.controls()
    assert isinstance(controls, Tabs)
    assert len(controls) == 2
    wb1, wb2 = controls
    assert isinstance(wb1, WidgetBox)
    assert len(wb1) == 7
    name, disabled, *ws = wb1
    assert isinstance(name, StaticText)
    assert isinstance(disabled, Checkbox)
    not_checked = []
    for w in ws:
        if w.name == 'Value':
            assert isinstance(w, TextInput)
            text_input.value = 'New value'
            assert w.value == 'New value'
        elif w.name == 'Value input':
            assert isinstance(w, TextInput)
        elif w.name == 'Placeholder':
            assert isinstance(w, TextInput)
            text_input.placeholder = 'Test placeholder...'
            assert w.value == 'Test placeholder...'
        elif w.name == 'Max length':
            assert isinstance(w, IntInput)
        elif w.name == 'Description':
            assert isinstance(w, TextInput)
            text_input.description = 'Test description...'
            assert w.value == 'Test description...'
        else:
            not_checked.append(w)
    assert not not_checked
    assert isinstance(wb2, WidgetBox)
    params1 = {w.name.replace(' ', '_').lower() for w in wb2 if len(w.name)}
    params2 = set(Viewable.param) - {'background', 'design', 'stylesheets', 'loading'}
    assert not len(params1 - params2)
    assert not len(params2 - params1)