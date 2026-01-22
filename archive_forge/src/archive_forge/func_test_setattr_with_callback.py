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
def test_setattr_with_callback(self):
    values = []
    options = OptionParser()
    options.define('foo', default=1, type=int, callback=values.append)
    options.foo = 2
    self.assertEqual(values, [2])