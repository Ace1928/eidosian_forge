import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
def wsgi_write_chunk(self, chunk):
    """
        Write a chunk of the output stream; send headers if they
        have not already been sent.
        """
    if not self.wsgi_headers_sent and (not self.wsgi_curr_headers):
        raise RuntimeError('Content returned before start_response called')
    if not self.wsgi_headers_sent:
        self.wsgi_headers_sent = True
        status, headers = self.wsgi_curr_headers
        code, message = status.split(' ', 1)
        self.send_response(int(code), message)
        send_close = True
        for k, v in headers:
            lk = k.lower()
            if 'content-length' == lk:
                send_close = False
            if 'connection' == lk:
                if 'close' == v.lower():
                    self.close_connection = 1
                    send_close = False
            self.send_header(k, v)
        if send_close:
            self.close_connection = 1
            self.send_header('Connection', 'close')
        self.end_headers()
    self.wfile.write(chunk)