from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_authorization_basic(header: str) -> Tuple[str, str]:
    """
    Parse an ``Authorization`` header for HTTP Basic Auth.

    Return a ``(username, password)`` tuple.

    Args:
        header: value of the ``Authorization`` header.

    Raises:
        InvalidHeaderFormat: on invalid inputs.
        InvalidHeaderValue: on unsupported inputs.

    """
    scheme, pos = parse_token(header, 0, 'Authorization')
    if scheme.lower() != 'basic':
        raise exceptions.InvalidHeaderValue('Authorization', f'unsupported scheme: {scheme}')
    if peek_ahead(header, pos) != ' ':
        raise exceptions.InvalidHeaderFormat('Authorization', 'expected space after scheme', header, pos)
    pos += 1
    basic_credentials, pos = parse_token68(header, pos, 'Authorization')
    parse_end(header, pos, 'Authorization')
    try:
        user_pass = base64.b64decode(basic_credentials.encode()).decode()
    except binascii.Error:
        raise exceptions.InvalidHeaderValue('Authorization', 'expected base64-encoded credentials') from None
    try:
        username, password = user_pass.split(':', 1)
    except ValueError:
        raise exceptions.InvalidHeaderValue('Authorization', 'expected username:password credentials') from None
    return (username, password)