from __future__ import annotations
import os
import typing
from http.client import HTTPException as HTTPException  # noqa: F401
from http.client import ResponseNotReady
from ..._base_connection import _TYPE_BODY
from ...connection import HTTPConnection, ProxyConfig, port_by_scheme
from ...exceptions import TimeoutError
from ...response import BaseHTTPResponse
from ...util.connection import _TYPE_SOCKET_OPTIONS
from ...util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from ...util.url import Url
from .fetch import _RequestError, _TimeoutError, send_request, send_streaming_request
from .request import EmscriptenRequest
from .response import EmscriptenHttpResponseWrapper, EmscriptenResponse
@property
def has_connected_to_proxy(self) -> bool:
    """Whether the connection has successfully connected to its proxy.
        This returns False if no proxy is in use. Used to determine whether
        errors are coming from the proxy layer or from tunnelling to the target origin.
        """
    return False