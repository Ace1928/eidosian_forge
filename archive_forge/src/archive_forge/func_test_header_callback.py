import base64
import binascii
from contextlib import closing
import copy
import gzip
import threading
import datetime
from io import BytesIO
import subprocess
import sys
import time
import typing  # noqa: F401
import unicodedata
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen
from tornado.httpclient import (
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado.log import gen_log, app_log
from tornado import netutil
from tornado.testing import AsyncHTTPTestCase, bind_unused_port, gen_test, ExpectLog
from tornado.test.util import skipOnTravis, ignore_deprecation
from tornado.web import Application, RequestHandler, url
from tornado.httputil import format_timestamp, HTTPHeaders
def test_header_callback(self):
    first_line = []
    headers = {}
    chunks = []

    def header_callback(header_line):
        if header_line.startswith('HTTP/1.1 101'):
            pass
        elif header_line.startswith('HTTP/'):
            first_line.append(header_line)
        elif header_line != '\r\n':
            k, v = header_line.split(':', 1)
            headers[k.lower()] = v.strip()

    def streaming_callback(chunk):
        self.assertEqual(headers['content-type'], 'text/html; charset=UTF-8')
        chunks.append(chunk)
    self.fetch('/chunk', header_callback=header_callback, streaming_callback=streaming_callback)
    self.assertEqual(len(first_line), 1, first_line)
    self.assertRegex(first_line[0], 'HTTP/[0-9]\\.[0-9] 200.*\r\n')
    self.assertEqual(chunks, [b'asdf', b'qwer'])