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
def test_name_swap(self):
    signed1 = create_signed_value(SignedValueTest.SECRET, 'key1', 'value', clock=self.present)
    signed2 = create_signed_value(SignedValueTest.SECRET, 'key2', 'value', clock=self.present)
    decoded1 = decode_signed_value(SignedValueTest.SECRET, 'key2', signed1, clock=self.present)
    self.assertIs(decoded1, None)
    decoded2 = decode_signed_value(SignedValueTest.SECRET, 'key1', signed2, clock=self.present)
    self.assertIs(decoded2, None)