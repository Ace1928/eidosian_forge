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
def test_cross_user(self):
    token2 = self.get_token()
    for token in (self.xsrf_token, token2):
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=token)), headers=self.cookie_headers(token))
        self.assertEqual(response.code, 200)
    for cookie_token, body_token in ((self.xsrf_token, token2), (token2, self.xsrf_token)):
        with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=body_token)), headers=self.cookie_headers(cookie_token))
        self.assertEqual(response.code, 403)