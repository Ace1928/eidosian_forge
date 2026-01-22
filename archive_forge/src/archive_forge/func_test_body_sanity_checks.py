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
def test_body_sanity_checks(self):
    for method in ('POST', 'PUT', 'PATCH'):
        with self.assertRaises(ValueError) as context:
            self.fetch('/all_methods', method=method, raise_error=True)
        self.assertIn('must not be None', str(context.exception))
        resp = self.fetch('/all_methods', method=method, allow_nonstandard_methods=True)
        self.assertEqual(resp.code, 200)
    for method in ('GET', 'DELETE', 'OPTIONS'):
        with self.assertRaises(ValueError) as context:
            self.fetch('/all_methods', method=method, body=b'asdf', raise_error=True)
        self.assertIn('must be None', str(context.exception))
        if method != 'GET':
            self.fetch('/all_methods', method=method, body=b'asdf', allow_nonstandard_methods=True, raise_error=True)
            self.assertEqual(resp.code, 200)