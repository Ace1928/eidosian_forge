import json
import os
import socket
import subprocess
import sys
import time
import uuid
from typing import Generator, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import pytest
from .conftest import KNOWN_SERVERS, extra_node_roots
@pytest.mark.parametrize('route', REST_ROUTES)
def test_auth_rest(route: str, a_server_url_and_token: Tuple[str, str]) -> None:
    """Verify a REST route only provides access to an authenticated user."""
    base_url, token = a_server_url_and_token
    verify_response(base_url, route)
    raw_body = verify_response(base_url, f'{route}?token={token}', 200)
    assert raw_body is not None, f'no response received from {route}'
    decode_error = None
    try:
        json.loads(raw_body.decode('utf-8'))
    except json.decoder.JSONDecodeError as err:
        decode_error = err
    assert not decode_error, f'the response for {route} was not JSON: {decode_error}'