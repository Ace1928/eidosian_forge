import unittest
from unittest.test.testmock.support import is_instance, X, SomeClass
from unittest.mock import (
def test_hierarchy(self):
    self.assertTrue(issubclass(MagicMock, Mock))
    self.assertTrue(issubclass(NonCallableMagicMock, NonCallableMock))