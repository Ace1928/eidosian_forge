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
def test_body_encoding(self):
    unicode_body = 'é'
    byte_body = binascii.a2b_hex(b'e9')
    response = self.fetch('/echopost', method='POST', body=unicode_body, headers={'Content-Type': 'application/blah'})
    self.assertEqual(response.headers['Content-Length'], '2')
    self.assertEqual(response.body, utf8(unicode_body))
    response = self.fetch('/echopost', method='POST', body=byte_body, headers={'Content-Type': 'application/blah'})
    self.assertEqual(response.headers['Content-Length'], '1')
    self.assertEqual(response.body, byte_body)
    response = self.fetch('/echopost', method='POST', body=byte_body, headers={'Content-Type': 'application/blah'}, user_agent='foo')
    self.assertEqual(response.headers['Content-Length'], '1')
    self.assertEqual(response.body, byte_body)