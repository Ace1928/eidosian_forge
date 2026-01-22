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
def log_startup(self) -> None:
    """Show information about the address when starting the server."""
    dev_warning = 'WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.'
    dev_warning = _ansi_style(dev_warning, 'bold', 'red')
    messages = [dev_warning]
    if self.address_family == af_unix:
        messages.append(f' * Running on {self.host}')
    else:
        scheme = 'http' if self.ssl_context is None else 'https'
        display_hostname = self.host
        if self.host in {'0.0.0.0', '::'}:
            messages.append(f' * Running on all addresses ({self.host})')
            if self.host == '0.0.0.0':
                localhost = '127.0.0.1'
                display_hostname = get_interface_ip(socket.AF_INET)
            else:
                localhost = '[::1]'
                display_hostname = get_interface_ip(socket.AF_INET6)
            messages.append(f' * Running on {scheme}://{localhost}:{self.port}')
        if ':' in display_hostname:
            display_hostname = f'[{display_hostname}]'
        messages.append(f' * Running on {scheme}://{display_hostname}:{self.port}')
    _log('info', '\n'.join(messages))