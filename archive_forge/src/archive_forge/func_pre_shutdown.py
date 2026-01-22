import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional  # noqa
from .abc import AbstractStreamWriter
from .helpers import get_running_loop
from .http_parser import RawRequestMessage
from .streams import StreamReader
from .web_protocol import RequestHandler, _RequestFactory, _RequestHandler
from .web_request import BaseRequest
def pre_shutdown(self) -> None:
    for conn in self._connections:
        conn.close()