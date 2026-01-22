from __future__ import annotations
import errno
import io
import os
import selectors
import socket
import socketserver
import sys
import typing as t
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.parse import unquote
from urllib.parse import urlsplit
from ._internal import _log
from ._internal import _wsgi_encoding_dance
from .exceptions import InternalServerError
from .urls import uri_to_iri
def read_chunk_len(self) -> int:
    try:
        line = self._rfile.readline().decode('latin1')
        _len = int(line.strip(), 16)
    except ValueError as e:
        raise OSError('Invalid chunk header') from e
    if _len < 0:
        raise OSError('Negative chunk length not allowed')
    return _len