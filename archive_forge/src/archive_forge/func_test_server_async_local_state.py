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
def test_server_async_local_state(bokeh_curdoc):
    docs = {}

    async def task():
        await asyncio.sleep(0.5)
        docs[state.curdoc] = []
        for _ in range(5):
            await asyncio.sleep(0.1)
            docs[state.curdoc].append(state.curdoc)

    def app():
        state.execute(task)
        return 'My app'
    serve_and_request(app, n=3)
    wait_until(lambda: len(docs) == 3)
    wait_until(lambda: all([len(set(docs)) == 1 and docs[0] is doc for doc, docs in docs.items()]))