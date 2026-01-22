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
def test_url_concat_dict_params(self):
    url = url_concat('https://localhost/path', dict(y='y'))
    self.assertEqual(url, 'https://localhost/path?y=y')