import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_url_escape_quote_plus(self):
    unescaped = '+ #%'
    plus_escaped = '%2B+%23%25'
    escaped = '%2B%20%23%25'
    self.assertEqual(url_escape(unescaped), plus_escaped)
    self.assertEqual(url_escape(unescaped, plus=False), escaped)
    self.assertEqual(url_unescape(plus_escaped), unescaped)
    self.assertEqual(url_unescape(escaped, plus=False), unescaped)
    self.assertEqual(url_unescape(plus_escaped, encoding=None), utf8(unescaped))
    self.assertEqual(url_unescape(escaped, encoding=None, plus=False), utf8(unescaped))