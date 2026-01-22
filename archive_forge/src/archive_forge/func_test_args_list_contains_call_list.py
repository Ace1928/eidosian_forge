import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_args_list_contains_call_list(self):
    mock = Mock()
    self.assertIsInstance(mock.call_args_list, _CallList)
    mock(1, 2)
    mock(a=3)
    mock(3, 4)
    mock(b=6)
    for kall in (call(1, 2), call(a=3), call(3, 4), call(b=6)):
        self.assertIn(kall, mock.call_args_list)
    calls = [call(a=3), call(3, 4)]
    self.assertIn(calls, mock.call_args_list)
    calls = [call(1, 2), call(a=3)]
    self.assertIn(calls, mock.call_args_list)
    calls = [call(3, 4), call(b=6)]
    self.assertIn(calls, mock.call_args_list)
    calls = [call(3, 4)]
    self.assertIn(calls, mock.call_args_list)
    self.assertNotIn(call('fish'), mock.call_args_list)
    self.assertNotIn([call('fish')], mock.call_args_list)