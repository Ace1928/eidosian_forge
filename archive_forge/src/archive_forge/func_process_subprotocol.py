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
@staticmethod
def process_subprotocol(headers: Headers, available_subprotocols: Optional[Sequence[Subprotocol]]) -> Optional[Subprotocol]:
    """
        Handle the Sec-WebSocket-Protocol HTTP response header.

        Check that it contains exactly one supported subprotocol.

        Return the selected subprotocol.

        """
    subprotocol: Optional[Subprotocol] = None
    header_values = headers.get_all('Sec-WebSocket-Protocol')
    if header_values:
        if available_subprotocols is None:
            raise InvalidHandshake('no subprotocols supported')
        parsed_header_values: Sequence[Subprotocol] = sum([parse_subprotocol(header_value) for header_value in header_values], [])
        if len(parsed_header_values) > 1:
            subprotocols = ', '.join(parsed_header_values)
            raise InvalidHandshake(f'multiple subprotocols: {subprotocols}')
        subprotocol = parsed_header_values[0]
        if subprotocol not in available_subprotocols:
            raise NegotiationError(f'unsupported subprotocol: {subprotocol}')
    return subprotocol