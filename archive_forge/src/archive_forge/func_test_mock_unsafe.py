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
def test_mock_unsafe(self):
    m = Mock()
    with self.assertRaises(AttributeError):
        m.assert_foo_call()
    with self.assertRaises(AttributeError):
        m.assret_foo_call()
    m = Mock(unsafe=True)
    m.assert_foo_call()
    m.assret_foo_call()