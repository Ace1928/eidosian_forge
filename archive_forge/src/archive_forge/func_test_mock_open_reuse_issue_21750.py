import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_mock_open_reuse_issue_21750(self):
    mocked_open = mock.mock_open(read_data='data')
    f1 = mocked_open('a-name')
    f1_data = f1.read()
    f2 = mocked_open('another-name')
    f2_data = f2.read()
    self.assertEqual(f1_data, f2_data)