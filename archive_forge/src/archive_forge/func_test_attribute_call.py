import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_attribute_call(self):
    self.assertEqual(call.foo(1), ('foo', (1,), {}))
    self.assertEqual(call.bar.baz(fish='eggs'), ('bar.baz', (), {'fish': 'eggs'}))
    mock = Mock()
    mock.foo(1, 2, 3)
    mock.bar.baz(a=3, b=6)
    self.assertEqual(mock.method_calls, [call.foo(1, 2, 3), call.bar.baz(a=3, b=6)])