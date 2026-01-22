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
def test_assert_any_call(self):
    mock = Mock()
    mock(1, 2)
    mock(a=3)
    mock(1, b=6)
    mock.assert_any_call(1, 2)
    mock.assert_any_call(a=3)
    mock.assert_any_call(1, b=6)
    self.assertRaises(AssertionError, mock.assert_any_call)
    self.assertRaises(AssertionError, mock.assert_any_call, 1, 3)
    self.assertRaises(AssertionError, mock.assert_any_call, a=4)