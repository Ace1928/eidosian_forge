import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_escape_return_types(self):
    self.assertEqual(type(xhtml_escape('foo')), str)
    self.assertEqual(type(xhtml_escape('foo')), unicode_type)