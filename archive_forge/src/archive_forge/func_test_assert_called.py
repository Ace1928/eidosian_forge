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
def test_assert_called(self):
    m = Mock()
    with self.assertRaises(AssertionError):
        m.hello.assert_called()
    m.hello()
    m.hello.assert_called()
    m.hello()
    m.hello.assert_called()