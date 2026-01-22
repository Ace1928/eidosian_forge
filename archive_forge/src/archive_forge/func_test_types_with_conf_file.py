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
def test_types_with_conf_file(self):
    for config_file_name in ('options_test_types.cfg', 'options_test_types_str.cfg'):
        options = self._define_options()
        options.parse_config_file(os.path.join(os.path.dirname(__file__), config_file_name))
        self._check_options_values(options)