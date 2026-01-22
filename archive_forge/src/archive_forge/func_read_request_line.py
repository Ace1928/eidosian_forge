import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
def read_request_line(self):
    """Read and parse first line of the HTTP request.

        Returns:
            bool: True if the request line is valid or False if it's malformed.

        """
    request_line = self.rfile.readline()
    self.started_request = True
    if not request_line:
        return False
    if request_line == CRLF:
        request_line = self.rfile.readline()
        if not request_line:
            return False
    if not request_line.endswith(CRLF):
        self.simple_response('400 Bad Request', 'HTTP requires CRLF terminators')
        return False
    try:
        method, uri, req_protocol = request_line.strip().split(SPACE, 2)
        if not req_protocol.startswith(b'HTTP/'):
            self.simple_response('400 Bad Request', 'Malformed Request-Line: bad protocol')
            return False
        rp = req_protocol[5:].split(b'.', 1)
        if len(rp) != 2:
            self.simple_response('400 Bad Request', 'Malformed Request-Line: bad version')
            return False
        rp = tuple(map(int, rp))
        if rp > (1, 1):
            self.simple_response('505 HTTP Version Not Supported', 'Cannot fulfill request')
            return False
    except (ValueError, IndexError):
        self.simple_response('400 Bad Request', 'Malformed Request-Line')
        return False
    self.uri = uri
    self.method = method.upper()
    if self.strict_mode and method != self.method:
        resp = 'Malformed method name: According to RFC 2616 (section 5.1.1) and its successors RFC 7230 (section 3.1.1) and RFC 7231 (section 4.1) method names are case-sensitive and uppercase.'
        self.simple_response('400 Bad Request', resp)
        return False
    try:
        scheme, authority, path, qs, fragment = urllib.parse.urlsplit(uri)
    except UnicodeError:
        self.simple_response('400 Bad Request', 'Malformed Request-URI')
        return False
    uri_is_absolute_form = scheme or authority
    if self.method == b'OPTIONS':
        path = uri if self.proxy_mode and uri_is_absolute_form else path
    elif self.method == b'CONNECT':
        if not self.proxy_mode:
            self.simple_response('405 Method Not Allowed')
            return False
        uri_split = urllib.parse.urlsplit(b''.join((b'//', uri)))
        _scheme, _authority, _path, _qs, _fragment = uri_split
        _port = EMPTY
        try:
            _port = uri_split.port
        except ValueError:
            pass
        invalid_path = _authority != uri or not _port or any((_scheme, _path, _qs, _fragment))
        if invalid_path:
            self.simple_response('400 Bad Request', 'Invalid path in Request-URI: request-target must match authority-form.')
            return False
        authority = path = _authority
        scheme = qs = fragment = EMPTY
    else:
        disallowed_absolute = self.strict_mode and (not self.proxy_mode) and uri_is_absolute_form
        if disallowed_absolute:
            'Absolute URI is only allowed within proxies.'
            self.simple_response('400 Bad Request', 'Absolute URI not allowed if server is not a proxy.')
            return False
        invalid_path = self.strict_mode and (not uri.startswith(FORWARD_SLASH)) and (not uri_is_absolute_form)
        if invalid_path:
            'Path should start with a forward slash.'
            resp = 'Invalid path in Request-URI: request-target must contain origin-form which starts with absolute-path (URI starting with a slash "/").'
            self.simple_response('400 Bad Request', resp)
            return False
        if fragment:
            self.simple_response('400 Bad Request', 'Illegal #fragment in Request-URI.')
            return False
        if path is None:
            self.simple_response('400 Bad Request', 'Invalid path in Request-URI.')
            return False
        try:
            atoms = [urllib.parse.unquote_to_bytes(x) for x in QUOTED_SLASH_REGEX.split(path)]
        except ValueError as ex:
            self.simple_response('400 Bad Request', ex.args[0])
            return False
        path = QUOTED_SLASH.join(atoms)
    if not path.startswith(FORWARD_SLASH):
        path = FORWARD_SLASH + path
    if scheme is not EMPTY:
        self.scheme = scheme
    self.authority = authority
    self.path = path
    self.qs = qs
    sp = (int(self.server.protocol[5]), int(self.server.protocol[7]))
    if sp[0] != rp[0]:
        self.simple_response('505 HTTP Version Not Supported')
        return False
    self.request_protocol = req_protocol
    self.response_protocol = 'HTTP/%s.%s' % min(rp, sp)
    return True