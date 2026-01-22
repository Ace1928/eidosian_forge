import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_call_list_str(self):
    mock = Mock()
    mock(1, 2)
    mock.foo(a=3)
    mock.foo.bar().baz('fish', cat='dog')
    expected = "[call(1, 2),\n call.foo(a=3),\n call.foo.bar(),\n call.foo.bar().baz('fish', cat='dog')]"
    self.assertEqual(str(mock.mock_calls), expected)