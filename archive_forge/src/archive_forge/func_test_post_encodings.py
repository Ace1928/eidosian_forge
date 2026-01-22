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
def test_post_encodings(self):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    uni_text = 'chinese: 张三'
    for enc in ('utf8', 'gbk'):
        for quote in (True, False):
            with self.subTest(enc=enc, quote=quote):
                bin_text = uni_text.encode(enc)
                if quote:
                    bin_text = urllib.parse.quote(bin_text).encode('ascii')
                response = self.fetch('/post_' + enc, method='POST', headers=headers, body=b'data=' + bin_text)
                self.assertEqual(json_decode(response.body), {'echo': uni_text})