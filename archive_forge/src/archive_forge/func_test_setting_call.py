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
def test_setting_call(self):
    mock = Mock()

    def __call__(self, a):
        return self._mock_call(a)
    type(mock).__call__ = __call__
    mock('one')
    mock.assert_called_with('one')
    self.assertRaises(TypeError, mock, 'one', 'two')