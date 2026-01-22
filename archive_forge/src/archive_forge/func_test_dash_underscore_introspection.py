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
def test_dash_underscore_introspection(self):
    options = OptionParser()
    options.define('with-dash', group='g')
    options.define('with_underscore', group='g')
    all_options = ['help', 'with-dash', 'with_underscore']
    self.assertEqual(sorted(options), all_options)
    self.assertEqual(sorted((k for k, v in options.items())), all_options)
    self.assertEqual(sorted(options.as_dict().keys()), all_options)
    self.assertEqual(sorted(options.group_dict('g')), ['with-dash', 'with_underscore'])
    buf = StringIO()
    options.print_help(buf)
    self.assertIn('--with-dash', buf.getvalue())
    self.assertIn('--with-underscore', buf.getvalue())