import asyncio
import atexit
import os
import pathlib
import re
import shutil
import signal
import socket
import tempfile
import time
import unittest
from contextlib import contextmanager
from subprocess import PIPE, Popen
import pandas as pd
import pytest
from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.io.doc import curdoc, set_curdoc as set_bkdoc
from pyviz_comms import Comm
from panel import config, serve
from panel.config import panel_extension
from panel.io.reload import (
from panel.io.state import set_curdoc, state
from panel.pane import HTML, Markdown
@pytest.fixture()
def markdown_server_session():
    port = 5051
    html = Markdown('#Title')
    server = serve(html, port=port, show=False, start=False)
    session = pull_session(session_id='Test', url='http://localhost:{:d}/'.format(server.port), io_loop=server.io_loop)
    yield (html, server, session, port)
    try:
        server.stop()
    except AssertionError:
        pass