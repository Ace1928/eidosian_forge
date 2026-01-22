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
def test_204_invalid_content_length(self):
    with ExpectLog(gen_log, '.*Response with code 204 should not have body', level=logging.INFO):
        with self.assertRaises(HTTPStreamClosedError):
            self.fetch('/?error=1', raise_error=True)
            if not self.http1:
                self.skipTest('requires HTTP/1.x')
            if self.http_client.configured_class != SimpleAsyncHTTPClient:
                self.skipTest('curl client accepts invalid headers')