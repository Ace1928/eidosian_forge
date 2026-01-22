from __future__ import annotations
import asyncio
import email.utils
import functools
import http
import inspect
import logging
import socket
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike, MultipleValuesError
from ..exceptions import (
from ..extensions import Extension, ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..protocol import State
from ..typing import ExtensionHeader, LoggerLike, Origin, StatusLike, Subprotocol
from .compatibility import asyncio_timeout
from .handshake import build_response, check_request
from .http import read_request
from .protocol import WebSocketCommonProtocol
def select_subprotocol(self, client_subprotocols: Sequence[Subprotocol], server_subprotocols: Sequence[Subprotocol]) -> Optional[Subprotocol]:
    """
        Pick a subprotocol among those supported by the client and the server.

        If several subprotocols are available, select the preferred subprotocol
        by giving equal weight to the preferences of the client and the server.

        If no subprotocol is available, proceed without a subprotocol.

        You may provide a ``select_subprotocol`` argument to :func:`serve` or
        :class:`WebSocketServerProtocol` to override this logic. For example,
        you could reject the handshake if the client doesn't support a
        particular subprotocol, rather than accept the handshake without that
        subprotocol.

        Args:
            client_subprotocols: list of subprotocols offered by the client.
            server_subprotocols: list of subprotocols available on the server.

        Returns:
            Optional[Subprotocol]: Selected subprotocol, if a common subprotocol
            was found.

            :obj:`None` to continue without a subprotocol.

        """
    if self._select_subprotocol is not None:
        return self._select_subprotocol(client_subprotocols, server_subprotocols)
    subprotocols = set(client_subprotocols) & set(server_subprotocols)
    if not subprotocols:
        return None
    return sorted(subprotocols, key=lambda p: client_subprotocols.index(p) + server_subprotocols.index(p))[0]