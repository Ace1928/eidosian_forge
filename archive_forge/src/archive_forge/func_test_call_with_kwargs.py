import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_call_with_kwargs(self):
    args = _Call(((), dict(a=3, b=4)))
    self.assertEqual(args, (dict(a=3, b=4),))
    self.assertEqual(args, ('foo', dict(a=3, b=4)))
    self.assertEqual(args, ('foo', (), dict(a=3, b=4)))
    self.assertEqual(args, ((), dict(a=3, b=4)))