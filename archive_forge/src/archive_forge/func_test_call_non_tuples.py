import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_call_non_tuples(self):
    kall = _Call(((1, 2, 3),))
    for value in (1, None, self, int):
        self.assertNotEqual(kall, value)
        self.assertFalse(kall == value)