from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
def test_host_matching(self):
    self.app.add_handlers('www.example.com', [('/foo', HostMatchingTest.Handler, {'reply': '[0]'})])
    self.app.add_handlers('www\\.example\\.com', [('/bar', HostMatchingTest.Handler, {'reply': '[1]'})])
    self.app.add_handlers('www.example.com', [('/baz', HostMatchingTest.Handler, {'reply': '[2]'})])
    self.app.add_handlers('www.e.*e.com', [('/baz', HostMatchingTest.Handler, {'reply': '[3]'})])
    response = self.fetch('/foo')
    self.assertEqual(response.body, b'wildcard')
    response = self.fetch('/bar')
    self.assertEqual(response.code, 404)
    response = self.fetch('/baz')
    self.assertEqual(response.code, 404)
    response = self.fetch('/foo', headers={'Host': 'www.example.com'})
    self.assertEqual(response.body, b'[0]')
    response = self.fetch('/bar', headers={'Host': 'www.example.com'})
    self.assertEqual(response.body, b'[1]')
    response = self.fetch('/baz', headers={'Host': 'www.example.com'})
    self.assertEqual(response.body, b'[2]')
    response = self.fetch('/baz', headers={'Host': 'www.exe.com'})
    self.assertEqual(response.body, b'[3]')