from tornado import gen, netutil
from tornado.escape import (
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body
from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO
import typing
@gen_test
def test_invalid_content_length(self):
    test_cases = [('alphabetic', 'foo'), ('leading plus', '+10'), ('internal underscore', '1_0')]
    for name, value in test_cases:
        with self.subTest(name=name), closing(IOStream(socket.socket())) as stream:
            with ExpectLog(gen_log, '.*Only integer Content-Length is allowed', level=logging.INFO):
                yield stream.connect(('127.0.0.1', self.get_http_port()))
                stream.write(utf8(textwrap.dedent(f'                            POST /echo HTTP/1.1\n                            Content-Length: {value}\n                            Connection: close\n\n                            1234567890\n                            ').replace('\n', '\r\n')))
                yield stream.read_until_close()