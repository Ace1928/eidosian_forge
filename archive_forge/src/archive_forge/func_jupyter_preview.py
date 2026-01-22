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
@pytest.fixture
def jupyter_preview(request):
    path = pathlib.Path(request.fspath.dirname)
    rel = path.relative_to(pathlib.Path(request.config.invocation_dir).absolute())
    return f'http://localhost:{JUPYTER_PORT}/panel-preview/render/{str(rel)}'