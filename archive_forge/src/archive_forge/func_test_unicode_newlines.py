from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
def test_unicode_newlines(self):
    newlines = ['\x1b', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029']
    for newline in newlines:
        for encoding in ['utf8', 'latin1']:
            try:
                try:
                    encoded = newline.encode(encoding)
                except UnicodeEncodeError:
                    continue
                data = b'Cookie: foo=' + encoded + b'bar'
                headers = HTTPHeaders.parse(native_str(data.decode('latin1')))
                expected = [('Cookie', 'foo=' + native_str(encoded.decode('latin1')) + 'bar')]
                self.assertEqual(expected, list(headers.get_all()))
            except Exception:
                gen_log.warning('failed while trying %r in %s', newline, encoding)
                raise