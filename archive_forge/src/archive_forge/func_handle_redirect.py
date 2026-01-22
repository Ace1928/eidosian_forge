from __future__ import annotations
import asyncio
import functools
import logging
import random
import urllib.parse
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike
from ..exceptions import (
from ..extensions import ClientExtensionFactory, Extension
from ..extensions.permessage_deflate import enable_client_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..typing import ExtensionHeader, LoggerLike, Origin, Subprotocol
from ..uri import WebSocketURI, parse_uri
from .compatibility import asyncio_timeout
from .handshake import build_request, check_response
from .http import read_response
from .protocol import WebSocketCommonProtocol
def handle_redirect(self, uri: str) -> None:
    old_uri = self._uri
    old_wsuri = self._wsuri
    new_uri = urllib.parse.urljoin(old_uri, uri)
    new_wsuri = parse_uri(new_uri)
    if old_wsuri.secure and (not new_wsuri.secure):
        raise SecurityError('redirect from WSS to WS')
    same_origin = old_wsuri.host == new_wsuri.host and old_wsuri.port == new_wsuri.port
    if not same_origin:
        factory = self._create_connection.args[0]
        factory = functools.partial(factory.func, *factory.args, **dict(factory.keywords, host=new_wsuri.host, port=new_wsuri.port))
        self._create_connection = functools.partial(self._create_connection.func, *(factory, new_wsuri.host, new_wsuri.port), **self._create_connection.keywords)
    self._uri = new_uri
    self._wsuri = new_wsuri