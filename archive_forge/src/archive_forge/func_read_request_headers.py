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
def read_request_headers(self):
    """Read ``self.rfile`` into ``self.inheaders``.

        Ref: :py:attr:`self.inheaders <HTTPRequest.outheaders>`.

        :returns: success status
        :rtype: bool
        """
    try:
        self.header_reader(self.rfile, self.inheaders)
    except ValueError as ex:
        self.simple_response('400 Bad Request', ex.args[0])
        return False
    mrbs = self.server.max_request_body_size
    try:
        cl = int(self.inheaders.get(b'Content-Length', 0))
    except ValueError:
        self.simple_response('400 Bad Request', 'Malformed Content-Length Header.')
        return False
    if mrbs and cl > mrbs:
        self.simple_response('413 Request Entity Too Large', 'The entity sent with the request exceeds the maximum allowed bytes.')
        return False
    if self.response_protocol == 'HTTP/1.1':
        if self.inheaders.get(b'Connection', b'') == b'close':
            self.close_connection = True
    elif self.inheaders.get(b'Connection', b'') != b'Keep-Alive':
        self.close_connection = True
    te = None
    if self.response_protocol == 'HTTP/1.1':
        te = self.inheaders.get(b'Transfer-Encoding')
        if te:
            te = [x.strip().lower() for x in te.split(b',') if x.strip()]
    self.chunked_read = False
    if te:
        for enc in te:
            if enc == b'chunked':
                self.chunked_read = True
            else:
                self.simple_response('501 Unimplemented')
                self.close_connection = True
                return False
    if self.inheaders.get(b'Expect', b'') == b'100-continue':
        msg = b''.join((self.server.protocol.encode('ascii'), SPACE, b'100 Continue', CRLF, CRLF))
        try:
            self.conn.wfile.write(msg)
        except socket.error as ex:
            if ex.args[0] not in errors.socket_errors_to_ignore:
                raise
    return True