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
def test_link_properties_server(document):

    class ReactiveLink(Reactive):
        text = param.String(default='A')
    obj = ReactiveLink()
    div = Div()
    obj._link_props(div, ['text'], document, div)
    assert 'text' in div._callbacks
    cb = div._callbacks['text'][0]
    assert isinstance(cb, partial)
    assert cb.args == (document, div.ref['id'], None)
    assert cb.func == obj._server_change