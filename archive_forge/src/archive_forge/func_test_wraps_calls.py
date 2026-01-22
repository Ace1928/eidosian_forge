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
def test_wraps_calls(self):
    real = Mock()
    mock = Mock(wraps=real)
    self.assertEqual(mock(), real())
    real.reset_mock()
    mock(1, 2, fish=3)
    real.assert_called_with(1, 2, fish=3)