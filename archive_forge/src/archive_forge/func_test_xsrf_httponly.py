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
def test_xsrf_httponly(self):
    response = self.fetch('/')
    self.assertIn('httponly;', response.headers['Set-Cookie'].lower())
    self.assertIn('expires=', response.headers['Set-Cookie'].lower())
    header = response.headers.get('Set-Cookie')
    assert header is not None
    match = re.match('.*; expires=(?P<expires>.+);.*', header)
    assert match is not None
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=2)
    header_expires = email.utils.parsedate_to_datetime(match.groupdict()['expires'])
    if header_expires.tzinfo is None:
        header_expires = header_expires.replace(tzinfo=datetime.timezone.utc)
    self.assertTrue(abs((expires - header_expires).total_seconds()) < 10)