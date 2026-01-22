from the server to the kernel.
from __future__ import annotations
import asyncio
import calendar
import datetime as dt
import inspect
import json
import logging
import os
import pathlib
import textwrap
import time
from queue import Empty
from typing import Any, Awaitable
from urllib.parse import urljoin
import tornado
from bokeh.embed.bundle import extension_dirs
from bokeh.protocol import Protocol
from bokeh.protocol.exceptions import ProtocolError
from bokeh.protocol.receiver import Receiver
from bokeh.server.tornado import DEFAULT_KEEP_ALIVE_MS
from bokeh.server.views.multi_root_static_handler import MultiRootStaticHandler
from bokeh.server.views.static_handler import StaticHandler
from bokeh.server.views.ws import WSHandler
from bokeh.util.token import (
from jupyter_server.base.handlers import JupyterHandler
from tornado.ioloop import PeriodicCallback
from tornado.web import StaticFileHandler
from ..config import config
from .resources import DIST_DIR, ERROR_TEMPLATE, _env
from .server import COMPONENT_PATH, ComponentResourceHandler
from .state import state
import os
import pathlib
import sys
from panel.io.jupyter_executor import PanelExecutor
def generate_executor(path: str, token: str, root_url: str) -> str:
    """
    Generates the code to instantiate a PanelExecutor that is to
    be be run on the kernel to start a server session.

    Arguments
    ---------
    path: str
       The path of the Panel application to execute.
    token: str
       The Bokeh JWT token containing the session_id, request arguments,
       headers and cookies.
    root_url: str
        The root_url the server is running on.

    Returns
    -------
    The code to be executed inside the kernel.
    """
    execute_template = _env.from_string(EXECUTION_TEMPLATE)
    return textwrap.dedent(execute_template.render(path=path, token=token, root_url=root_url))