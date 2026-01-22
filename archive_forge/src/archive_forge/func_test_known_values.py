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
def test_known_values(self):
    signed_v1 = create_signed_value(SignedValueTest.SECRET, 'key', 'value', version=1, clock=self.present)
    self.assertEqual(signed_v1, b'dmFsdWU=|1300000000|31c934969f53e48164c50768b40cbd7e2daaaa4f')
    signed_v2 = create_signed_value(SignedValueTest.SECRET, 'key', 'value', version=2, clock=self.present)
    self.assertEqual(signed_v2, b'2|1:0|10:1300000000|3:key|8:dmFsdWU=|3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152')
    signed_default = create_signed_value(SignedValueTest.SECRET, 'key', 'value', clock=self.present)
    self.assertEqual(signed_default, signed_v2)
    decoded_v1 = decode_signed_value(SignedValueTest.SECRET, 'key', signed_v1, min_version=1, clock=self.present)
    self.assertEqual(decoded_v1, b'value')
    decoded_v2 = decode_signed_value(SignedValueTest.SECRET, 'key', signed_v2, min_version=2, clock=self.present)
    self.assertEqual(decoded_v2, b'value')