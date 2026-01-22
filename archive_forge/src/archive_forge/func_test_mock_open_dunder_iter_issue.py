import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_open_dunder_iter_issue(self):
    mocked_open = mock.mock_open(read_data='Remarkable\nNorwegian Blue')
    f1 = mocked_open('a-name')
    lines = [line for line in f1]
    self.assertEqual(lines[0], 'Remarkable\n')
    self.assertEqual(lines[1], 'Norwegian Blue')
    self.assertEqual(list(f1), [])