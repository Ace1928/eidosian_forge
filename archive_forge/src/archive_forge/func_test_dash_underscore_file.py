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
def test_dash_underscore_file(self):
    for defined_name in ['foo-bar', 'foo_bar']:
        options = OptionParser()
        options.define(defined_name)
        options.parse_config_file(os.path.join(os.path.dirname(__file__), 'options_test.cfg'))
        self.assertEqual(options.foo_bar, 'a')