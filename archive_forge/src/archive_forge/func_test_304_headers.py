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
def test_304_headers(self):
    response1 = self.fetch('/')
    self.assertEqual(response1.headers['Content-Length'], '5')
    self.assertEqual(response1.headers['Content-Language'], 'en_US')
    response2 = self.fetch('/', headers={'If-None-Match': response1.headers['Etag']})
    self.assertEqual(response2.code, 304)
    self.assertTrue('Content-Length' not in response2.headers)
    self.assertTrue('Content-Language' not in response2.headers)
    self.assertTrue('Transfer-Encoding' not in response2.headers)