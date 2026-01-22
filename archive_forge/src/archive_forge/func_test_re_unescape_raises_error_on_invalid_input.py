from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
def test_re_unescape_raises_error_on_invalid_input(self):
    with self.assertRaises(ValueError):
        re_unescape('\\d')
    with self.assertRaises(ValueError):
        re_unescape('\\b')
    with self.assertRaises(ValueError):
        re_unescape('\\Z')