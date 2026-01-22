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
def test_parse_command_line(self):
    options = OptionParser()
    options.define('port', default=80)
    options.parse_command_line(['main.py', '--port=443'])
    self.assertEqual(options.port, 443)