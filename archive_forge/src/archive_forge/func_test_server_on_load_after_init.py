import asyncio
import datetime as dt
import logging
import os
import pathlib
import time
import weakref
import param
import pytest
import requests
from bokeh.events import ButtonClick
from panel.config import config
from panel.io import state
from panel.io.resources import DIST_DIR, JS_VERSION
from panel.io.server import INDEX_HTML, get_server, set_curdoc
from panel.layout import Row
from panel.models import HTML as BkHTML
from panel.models.tabulator import TableEditEvent
from panel.pane import Markdown
from panel.param import ParamFunction
from panel.reactive import ReactiveHTML
from panel.template import BootstrapTemplate
from panel.tests.util import serve_and_request, serve_and_wait, wait_until
from panel.widgets import (
def test_server_on_load_after_init():
    loaded = []

    def cb():
        loaded.append((state.curdoc, state.loaded))

    def app():
        state.onload(cb)

        def loaded():
            state._schedule_on_load(state.curdoc, None)
        state.execute(loaded, schedule=True)
        return 'App'
    serve_and_request(app)
    wait_until(lambda: len(loaded) == 1)
    doc = loaded[0][0]
    with set_curdoc(doc):
        state.onload(cb)
    wait_until(lambda: len(loaded) == 2)
    assert loaded == [(doc, False), (doc, True)]