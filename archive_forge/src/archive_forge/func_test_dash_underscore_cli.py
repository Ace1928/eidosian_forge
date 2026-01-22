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
def test_dash_underscore_cli(self):
    for defined_name in ['foo-bar', 'foo_bar']:
        for flag in ['--foo-bar=a', '--foo_bar=a']:
            options = OptionParser()
            options.define(defined_name)
            options.parse_command_line(['main.py', flag])
            self.assertEqual(options.foo_bar, 'a')
            self.assertEqual(options['foo-bar'], 'a')
            self.assertEqual(options['foo_bar'], 'a')