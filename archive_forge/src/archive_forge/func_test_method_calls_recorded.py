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
def test_method_calls_recorded(self):
    mock = Mock()
    mock.something(3, fish=None)
    mock.something_else.something(6, cake=sentinel.Cake)
    self.assertEqual(mock.something_else.method_calls, [('something', (6,), {'cake': sentinel.Cake})], 'method calls not recorded correctly')
    self.assertEqual(mock.method_calls, [('something', (3,), {'fish': None}), ('something_else.something', (6,), {'cake': sentinel.Cake})], 'method calls not recorded correctly')