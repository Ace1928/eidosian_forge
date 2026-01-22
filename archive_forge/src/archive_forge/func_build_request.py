from __future__ import annotations
import base64
import binascii
from typing import List
from ..datastructures import Headers, MultipleValuesError
from ..exceptions import InvalidHeader, InvalidHeaderValue, InvalidUpgrade
from ..headers import parse_connection, parse_upgrade
from ..typing import ConnectionOption, UpgradeProtocol
from ..utils import accept_key as accept, generate_key
def build_request(headers: Headers) -> str:
    """
    Build a handshake request to send to the server.

    Update request headers passed in argument.

    Args:
        headers: Handshake request headers.

    Returns:
        str: ``key`` that must be passed to :func:`check_response`.

    """
    key = generate_key()
    headers['Upgrade'] = 'websocket'
    headers['Connection'] = 'Upgrade'
    headers['Sec-WebSocket-Key'] = key
    headers['Sec-WebSocket-Version'] = '13'
    return key