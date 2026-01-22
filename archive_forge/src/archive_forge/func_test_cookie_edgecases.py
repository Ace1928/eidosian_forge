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
def test_cookie_edgecases(self):
    self.assertEqual(parse_cookie('a=b; Domain=example.com'), {'a': 'b', 'Domain': 'example.com'})
    self.assertEqual(parse_cookie('a=b; h=i; a=c'), {'a': 'c', 'h': 'i'})