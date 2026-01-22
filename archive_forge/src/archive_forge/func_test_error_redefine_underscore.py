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
def test_error_redefine_underscore(self):
    tests = [('foo-bar', 'foo-bar'), ('foo_bar', 'foo_bar'), ('foo-bar', 'foo_bar'), ('foo_bar', 'foo-bar')]
    for a, b in tests:
        with subTest(self, a=a, b=b):
            options = OptionParser()
            options.define(a)
            with self.assertRaises(Error) as cm:
                options.define(b)
            self.assertRegex(str(cm.exception), 'Option.*foo.bar.*already defined')