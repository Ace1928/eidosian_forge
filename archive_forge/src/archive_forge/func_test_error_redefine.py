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
def test_error_redefine(self):
    options = OptionParser()
    options.define('foo')
    with self.assertRaises(Error) as cm:
        options.define('foo')
    self.assertRegex(str(cm.exception), 'Option.*foo.*already defined')