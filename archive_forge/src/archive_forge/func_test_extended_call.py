import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_extended_call(self):
    result = call(1).foo(2).bar(3, a=4)
    self.assertEqual(result, ('().foo().bar', (3,), dict(a=4)))
    mock = MagicMock()
    mock(1, 2, a=3, b=4)
    self.assertEqual(mock.call_args, call(1, 2, a=3, b=4))
    self.assertNotEqual(mock.call_args, call(1, 2, 3))
    self.assertEqual(mock.call_args_list, [call(1, 2, a=3, b=4)])
    self.assertEqual(mock.mock_calls, [call(1, 2, a=3, b=4)])
    mock = MagicMock()
    mock.foo(1).bar()().baz.beep(a=6)
    last_call = call.foo(1).bar()().baz.beep(a=6)
    self.assertEqual(mock.mock_calls[-1], last_call)
    self.assertEqual(mock.mock_calls, last_call.call_list())