import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_linkify(self):
    for text, kwargs, html in linkify_tests:
        linked = tornado.escape.linkify(text, **kwargs)
        self.assertEqual(linked, html)