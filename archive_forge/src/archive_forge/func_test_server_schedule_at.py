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
def test_server_schedule_at():

    def periodic_cb():
        state.cache['at'] = dt.datetime.now()
    scheduled = dt.datetime.now() + dt.timedelta(seconds=1.57)

    def app():
        state.schedule_task('periodic', periodic_cb, at=scheduled)
        return '# state.schedule test'
    serve_and_request(app)
    wait_until(lambda: 'at' in state.cache)
    assert abs(state.cache['at'] - scheduled) < dt.timedelta(seconds=0.2)
    assert len(state._scheduled) == 0