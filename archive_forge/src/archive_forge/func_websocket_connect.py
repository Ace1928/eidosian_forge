from __future__ import annotations
import contextlib
import inspect
import io
import json
import math
import queue
import sys
import typing
import warnings
from concurrent.futures import Future
from functools import cached_property
from types import GeneratorType
from urllib.parse import unquote, urljoin
import anyio
import anyio.abc
import anyio.from_thread
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from anyio.streams.stapled import StapledObjectStream
from starlette._utils import is_async_callable
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect
def websocket_connect(self, url: str, subprotocols: typing.Sequence[str] | None=None, **kwargs: typing.Any) -> WebSocketTestSession:
    url = urljoin('ws://testserver', url)
    headers = kwargs.get('headers', {})
    headers.setdefault('connection', 'upgrade')
    headers.setdefault('sec-websocket-key', 'testserver==')
    headers.setdefault('sec-websocket-version', '13')
    if subprotocols is not None:
        headers.setdefault('sec-websocket-protocol', ', '.join(subprotocols))
    kwargs['headers'] = headers
    try:
        super().request('GET', url, **kwargs)
    except _Upgrade as exc:
        session = exc.session
    else:
        raise RuntimeError('Expected WebSocket upgrade')
    return session