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
def get_and_head(self, *args, **kwargs):
    """Performs a GET and HEAD request and returns the GET response.

        Fails if any ``Content-*`` headers returned by the two requests
        differ.
        """
    head_response = self.fetch(*args, method='HEAD', **kwargs)
    get_response = self.fetch(*args, method='GET', **kwargs)
    content_headers = set()
    for h in itertools.chain(head_response.headers, get_response.headers):
        if h.startswith('Content-'):
            content_headers.add(h)
    for h in content_headers:
        self.assertEqual(head_response.headers.get(h), get_response.headers.get(h), '%s differs between GET (%s) and HEAD (%s)' % (h, head_response.headers.get(h), get_response.headers.get(h)))
    return get_response