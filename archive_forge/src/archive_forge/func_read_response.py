import asyncio
import socket
import sys
from typing import Callable, List, Optional, Union
from redis.compat import TypedDict
from ..exceptions import ConnectionError, InvalidResponse, RedisError
from ..typing import EncodableT
from ..utils import HIREDIS_AVAILABLE
from .base import AsyncBaseParser, BaseParser
from .socket import (
def read_response(self, disable_decoding=False):
    if not self._reader:
        raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
    if self._next_response is not False:
        response = self._next_response
        self._next_response = False
        return response
    if disable_decoding:
        response = self._reader.gets(False)
    else:
        response = self._reader.gets()
    while response is False:
        self.read_from_socket()
        if disable_decoding:
            response = self._reader.gets(False)
        else:
            response = self._reader.gets()
    if isinstance(response, ConnectionError):
        raise response
    elif isinstance(response, list) and response and isinstance(response[0], ConnectionError):
        raise response[0]
    return response