import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_open_using_next(self):
    mocked_open = mock.mock_open(read_data='1st line\n2nd line\n3rd line')
    f1 = mocked_open('a-name')
    line1 = next(f1)
    line2 = f1.__next__()
    lines = [line for line in f1]
    self.assertEqual(line1, '1st line\n')
    self.assertEqual(line2, '2nd line\n')
    self.assertEqual(lines[0], '3rd line')
    self.assertEqual(list(f1), [])
    with self.assertRaises(StopIteration):
        next(f1)