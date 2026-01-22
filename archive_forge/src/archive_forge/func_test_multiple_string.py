import datetime
from io import StringIO
import os
import sys
from unittest import mock
import unittest
from tornado.options import OptionParser, Error
from tornado.util import basestring_type
from tornado.test.util import subTest
import typing
def test_multiple_string(self):
    options = OptionParser()
    options.define('foo', type=str, multiple=True)
    options.parse_command_line(['main.py', '--foo=a,b,c'])
    self.assertEqual(options.foo, ['a', 'b', 'c'])