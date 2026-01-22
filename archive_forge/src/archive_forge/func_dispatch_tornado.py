from __future__ import annotations
import asyncio
import dataclasses
import datetime as dt
import gc
import inspect
import json
import logging
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
from bokeh.application.application import SessionContext
from bokeh.core.serialization import Serializable
from bokeh.document.document import Document
from bokeh.document.events import (
from bokeh.model.util import visit_immediate_value_references
from bokeh.models import CustomJS
from ..config import config
from ..util import param_watchers
from .loading import LOADING_INDICATOR_CSS_CLASS
from .model import hold, monkeypatch_events  # noqa: F401 API import
from .state import curdoc_locked, state
def dispatch_tornado(conn, events=None, msg=None):
    from tornado.websocket import WebSocketHandler
    socket = conn._socket
    ws_conn = getattr(socket, 'ws_connection', False)
    if not ws_conn or ws_conn.is_closing():
        return []
    if msg is None:
        msg = conn.protocol.create('PATCH-DOC', events)
    futures = [WebSocketHandler.write_message(socket, msg.header_json), WebSocketHandler.write_message(socket, msg.metadata_json), WebSocketHandler.write_message(socket, msg.content_json)]
    for buffer in msg._buffers:
        header = json.dumps(buffer.ref)
        payload = buffer.to_bytes()
        futures.extend([WebSocketHandler.write_message(socket, header), WebSocketHandler.write_message(socket, payload, binary=True)])
    return futures