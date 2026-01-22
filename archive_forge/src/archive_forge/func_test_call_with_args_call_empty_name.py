import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_call_with_args_call_empty_name(self):
    args = _Call(((1, 2, 3), {}))
    self.assertEqual(args, call(1, 2, 3))
    self.assertEqual(call(1, 2, 3), args)
    self.assertIn(call(1, 2, 3), [args])