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
def test_serve_config_per_session_state():
    CSS1 = 'body { background-color: red }'
    CSS2 = 'body { background-color: green }'

    def app1():
        config.raw_css = [CSS1]

    def app2():
        config.raw_css = [CSS2]
    port1, port2 = (7001, 7002)
    serve_and_wait(app1, port=port1)
    serve_and_wait(app2, port=port2)
    r1 = requests.get(f'http://localhost:{port1}/').content.decode('utf-8')
    r2 = requests.get(f'http://localhost:{port2}/').content.decode('utf-8')
    assert CSS1 not in config.raw_css
    assert CSS2 not in config.raw_css
    assert CSS1 in r1
    assert CSS2 not in r1
    assert CSS1 not in r2
    assert CSS2 in r2