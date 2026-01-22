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
def test_mock_patch(self):
    options = OptionParser()
    options.define('foo', default=1)
    options.parse_command_line(['main.py', '--foo=2'])
    self.assertEqual(options.foo, 2)
    with mock.patch.object(options.mockable(), 'foo', 3):
        self.assertEqual(options.foo, 3)
    self.assertEqual(options.foo, 2)
    with mock.patch.object(options.mockable(), 'foo', 4):
        self.assertEqual(options.foo, 4)
        options.foo = 5
        self.assertEqual(options.foo, 5)
        with mock.patch.object(options.mockable(), 'foo', 6):
            self.assertEqual(options.foo, 6)
        self.assertEqual(options.foo, 5)
    self.assertEqual(options.foo, 2)