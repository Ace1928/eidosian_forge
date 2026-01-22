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
def process_extensions(headers: Headers, available_extensions: Optional[Sequence[ClientExtensionFactory]]) -> List[Extension]:
    """
        Handle the Sec-WebSocket-Extensions HTTP response header.

        Check that each extension is supported, as well as its parameters.

        Return the list of accepted extensions.

        Raise :exc:`~websockets.exceptions.InvalidHandshake` to abort the
        connection.

        :rfc:`6455` leaves the rules up to the specification of each
        :extension.

        To provide this level of flexibility, for each extension accepted by
        the server, we check for a match with each extension available in the
        client configuration. If no match is found, an exception is raised.

        If several variants of the same extension are accepted by the server,
        it may be configured several times, which won't make sense in general.
        Extensions must implement their own requirements. For this purpose,
        the list of previously accepted extensions is provided.

        Other requirements, for example related to mandatory extensions or the
        order of extensions, may be implemented by overriding this method.

        """
    accepted_extensions: List[Extension] = []
    header_values = headers.get_all('Sec-WebSocket-Extensions')
    if header_values:
        if available_extensions is None:
            raise InvalidHandshake('no extensions supported')
        parsed_header_values: List[ExtensionHeader] = sum([parse_extension(header_value) for header_value in header_values], [])
        for name, response_params in parsed_header_values:
            for extension_factory in available_extensions:
                if extension_factory.name != name:
                    continue
                try:
                    extension = extension_factory.process_response_params(response_params, accepted_extensions)
                except NegotiationError:
                    continue
                accepted_extensions.append(extension)
                break
            else:
                raise NegotiationError(f'Unsupported extension: name = {name}, params = {response_params}')
    return accepted_extensions