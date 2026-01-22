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
def test_ip_headers(self):
    self.assertEqual(self.fetch_json('/')['remote_ip'], '127.0.0.1')
    valid_ipv4 = {'X-Real-IP': '4.4.4.4'}
    self.assertEqual(self.fetch_json('/', headers=valid_ipv4)['remote_ip'], '4.4.4.4')
    valid_ipv4_list = {'X-Forwarded-For': '127.0.0.1, 4.4.4.4'}
    self.assertEqual(self.fetch_json('/', headers=valid_ipv4_list)['remote_ip'], '4.4.4.4')
    valid_ipv6 = {'X-Real-IP': '2620:0:1cfe:face:b00c::3'}
    self.assertEqual(self.fetch_json('/', headers=valid_ipv6)['remote_ip'], '2620:0:1cfe:face:b00c::3')
    valid_ipv6_list = {'X-Forwarded-For': '::1, 2620:0:1cfe:face:b00c::3'}
    self.assertEqual(self.fetch_json('/', headers=valid_ipv6_list)['remote_ip'], '2620:0:1cfe:face:b00c::3')
    invalid_chars = {'X-Real-IP': '4.4.4.4<script>'}
    self.assertEqual(self.fetch_json('/', headers=invalid_chars)['remote_ip'], '127.0.0.1')
    invalid_chars_list = {'X-Forwarded-For': '4.4.4.4, 5.5.5.5<script>'}
    self.assertEqual(self.fetch_json('/', headers=invalid_chars_list)['remote_ip'], '127.0.0.1')
    invalid_host = {'X-Real-IP': 'www.google.com'}
    self.assertEqual(self.fetch_json('/', headers=invalid_host)['remote_ip'], '127.0.0.1')