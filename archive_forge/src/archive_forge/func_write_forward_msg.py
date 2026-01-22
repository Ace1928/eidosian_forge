from __future__ import annotations
import base64
import binascii
import json
from typing import Any, Awaitable, Final
import tornado.concurrent
import tornado.locks
import tornado.netutil
import tornado.web
import tornado.websocket
from tornado.websocket import WebSocketHandler
from streamlit import config
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import Runtime, SessionClient, SessionClientDisconnectedError
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import is_url_from_allowed_origins
def write_forward_msg(self, msg: ForwardMsg) -> None:
    """Send a ForwardMsg to the browser."""
    try:
        self.write_message(serialize_forward_msg(msg), binary=True)
    except tornado.websocket.WebSocketClosedError as e:
        raise SessionClientDisconnectedError from e