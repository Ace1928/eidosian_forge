import base64
from contextlib import closing
import gzip
from http.server import BaseHTTPRequestHandler
import os
import socket
from socketserver import ThreadingMixIn
import ssl
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import (
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer
from .openmetrics import exposition as openmetrics
from .registry import CollectorRegistry, REGISTRY
from .utils import floatToGoString
from .asgi import make_asgi_app  # noqa
def basic_auth_handler(url: str, method: str, timeout: Optional[float], headers: List[Tuple[str, str]], data: bytes, username: Optional[str]=None, password: Optional[str]=None) -> Callable[[], None]:
    """Handler that implements HTTP/HTTPS connections with Basic Auth.

    Sets auth headers using supplied 'username' and 'password', if set.
    Used by the push_to_gateway functions. Can be re-used by other handlers."""

    def handle():
        """Handler that implements HTTP Basic Auth.
        """
        if username is not None and password is not None:
            auth_value = f'{username}:{password}'.encode()
            auth_token = base64.b64encode(auth_value)
            auth_header = b'Basic ' + auth_token
            headers.append(('Authorization', auth_header))
        default_handler(url, method, timeout, headers, data)()
    return handle