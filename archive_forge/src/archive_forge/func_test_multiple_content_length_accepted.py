import collections
from contextlib import closing
import errno
import logging
import os
import re
import socket
import ssl
import sys
import typing  # noqa: F401
from tornado.escape import to_unicode, utf8
from tornado import gen, version
from tornado.httpclient import AsyncHTTPClient
from tornado.httputil import HTTPHeaders, ResponseStartLine
from tornado.ioloop import IOLoop
from tornado.iostream import UnsatisfiableReadError
from tornado.locks import Event
from tornado.log import gen_log
from tornado.netutil import Resolver, bind_sockets
from tornado.simple_httpclient import (
from tornado.test.httpclient_test import (
from tornado.test import httpclient_test
from tornado.testing import (
from tornado.test.util import skipOnTravis, skipIfNoIPv6, refusing_port
from tornado.web import RequestHandler, Application, url, stream_request_body
def test_multiple_content_length_accepted(self: typing.Any):
    response = self.fetch('/content_length?value=2,2')
    self.assertEqual(response.body, b'ok')
    response = self.fetch('/content_length?value=2,%202,2')
    self.assertEqual(response.body, b'ok')
    with ExpectLog(gen_log, '.*Multiple unequal Content-Lengths', level=logging.INFO):
        with self.assertRaises(HTTPStreamClosedError):
            self.fetch('/content_length?value=2,4', raise_error=True)
        with self.assertRaises(HTTPStreamClosedError):
            self.fetch('/content_length?value=2,%202,3', raise_error=True)