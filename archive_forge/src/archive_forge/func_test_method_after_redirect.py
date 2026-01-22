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
def test_method_after_redirect(self):
    for status in [301, 302, 303]:
        url = '/redirect?url=/all_methods&status=%d' % status
        resp = self.fetch(url, method='POST', body=b'')
        self.assertEqual(b'GET', resp.body)
        for method in ['GET', 'OPTIONS', 'PUT', 'DELETE']:
            resp = self.fetch(url, method=method, allow_nonstandard_methods=True)
            if status in [301, 302]:
                self.assertEqual(utf8(method), resp.body)
            else:
                self.assertIn(resp.body, [utf8(method), b'GET'])
        resp = self.fetch(url, method='HEAD')
        self.assertEqual(200, resp.code)
        self.assertEqual(b'', resp.body)
    for status in [307, 308]:
        url = '/redirect?url=/all_methods&status=307'
        for method in ['GET', 'OPTIONS', 'POST', 'PUT', 'DELETE']:
            resp = self.fetch(url, method=method, allow_nonstandard_methods=True)
            self.assertEqual(method, to_unicode(resp.body))
        resp = self.fetch(url, method='HEAD')
        self.assertEqual(200, resp.code)
        self.assertEqual(b'', resp.body)