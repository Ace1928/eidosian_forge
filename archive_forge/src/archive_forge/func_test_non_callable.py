import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_non_callable(self):
    for mock in (NonCallableMagicMock(), NonCallableMock()):
        self.assertRaises(TypeError, mock)
        self.assertFalse(hasattr(mock, '__call__'))
        self.assertIn(mock.__class__.__name__, repr(mock))