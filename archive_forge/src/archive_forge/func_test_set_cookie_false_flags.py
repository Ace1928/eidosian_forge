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
def test_set_cookie_false_flags(self):
    response = self.fetch('/set_falsy_flags')
    headers = sorted(response.headers.get_list('Set-Cookie'))
    self.assertEqual(headers[0].lower(), 'a=1; path=/; secure')
    self.assertEqual(headers[1].lower(), 'b=1; path=/')
    self.assertEqual(headers[2].lower(), 'c=1; httponly; path=/')
    self.assertEqual(headers[3].lower(), 'd=1; path=/')