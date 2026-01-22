import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_open_next_with_readline_with_return_value(self):
    mopen = mock.mock_open(read_data='foo\nbarn')
    mopen.return_value.readline.return_value = 'abc'
    self.assertEqual('abc', next(mopen()))